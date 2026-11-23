"""Tkinter desktop app: choose a photo, run the approved model and save RGB output."""
from pathlib import Path
import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from PIL import Image,ImageTk
from ..application.enhancement import EnhancementService
from kind_enhance.inference.pipeline import ROOT
from ..infrastructure.images import read_photo,prepare_photo,save_png


def configure_display():
    """Use physical Windows coordinates so the app fits high-DPI desktops."""
    if sys.platform!='win32':return
    import ctypes
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError,OSError):
        try:ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError,OSError):pass


class EnhancementApp:
    def __init__(self,root,service=None):
        self.root=root
        self.service=service or EnhancementService()
        self.selected_path=None
        self.result=None
        self.last_output_path=None
        self.busy=False
        self._closing=False
        self._events=queue.Queue()
        self._input_image=None
        self._output_image=None
        self._photos=[]
        self._render_job=None
        root.title('KinD | Tăng sáng ảnh RGB')
        width=min(1180,max(720,root.winfo_screenwidth()-80))
        height=min(760,max(520,root.winfo_screenheight()-120))
        root.geometry(f'{width}x{height}')
        root.minsize(min(940,width),min(640,height))
        root.configure(background='#eef2f7')
        root.protocol('WM_DELETE_WINDOW',self.close)
        self._build()
        self._poll_job=root.after(100,self._poll)
        root.bind('<Control-o>',lambda event:self.choose_image())
        root.bind('<Control-s>',lambda event:self.save_copy())
        root.bind('<Configure>',self._schedule_render)

    def _build(self):
        style=ttk.Style(self.root)
        if 'clam' in style.theme_names():style.theme_use('clam')
        style.configure('TButton',font=('Segoe UI',10),padding=(14,9))
        style.configure('Primary.TButton',background='#2563eb',foreground='white')
        style.map('Primary.TButton',background=[('active','#1d4ed8'),('disabled','#c3d3ee')])
        style.configure('TCombobox',padding=5)
        self.root.columnconfigure(0,weight=1)
        self.root.rowconfigure(2,weight=1)
        header=tk.Frame(self.root,bg='#14243b',padx=24,pady=20)
        header.grid(row=0,column=0,sticky='ew')
        tk.Label(header,text='TĂNG SÁNG ẢNH',font=('Segoe UI',20,'bold'),bg='#14243b',fg='white').pack(anchor='w')
        tk.Label(header,text='Chọn ảnh tối · Xem kết quả · Lưu ảnh RGB',font=('Segoe UI',11),bg='#14243b',fg='#b9c9df').pack(anchor='w',pady=(4,0))
        toolbar=tk.Frame(self.root,bg='#eef2f7',padx=20,pady=16)
        toolbar.grid(row=1,column=0,sticky='ew')
        self.choose_button=ttk.Button(toolbar,text='1. Chọn ảnh',command=self.choose_image)
        self.choose_button.pack(side='left')
        self.run_button=ttk.Button(toolbar,text='2. Tăng sáng',style='Primary.TButton',command=self.start_processing,state='disabled')
        self.run_button.pack(side='left',padx=10)
        self.save_button=ttk.Button(toolbar,text='Lưu bản sao...',command=self.save_copy,state='disabled')
        self.save_button.pack(side='left')
        self.folder_button=ttk.Button(toolbar,text='Mở kết quả',command=self.open_folder,state='disabled')
        self.folder_button.pack(side='left',padx=10)
        tk.Label(toolbar,text='Cạnh tối đa:',font=('Segoe UI',10),bg='#eef2f7',fg='#42536b').pack(side='left',padx=(12,5))
        self.resolution=ttk.Combobox(toolbar,values=('800','1280','1600'),state='readonly',width=6)
        self.resolution.set('1280')
        self.resolution.pack(side='left')
        self.resolution.bind('<<ComboboxSelected>>',self._resolution_changed)
        workspace=tk.Frame(self.root,bg='#eef2f7',padx=20)
        workspace.grid(row=2,column=0,sticky='nsew')
        workspace.columnconfigure(0,weight=1,uniform='preview')
        workspace.columnconfigure(1,weight=1,uniform='preview')
        workspace.rowconfigure(0,weight=1)
        self.input_preview,self.input_caption=self._panel(workspace,0,'Ảnh đầu vào','Chọn một ảnh tối để bắt đầu')
        self.output_preview,self.output_caption=self._panel(workspace,1,'Ảnh tăng sáng (RGB)','Kết quả sẽ xuất hiện ở đây')
        footer=tk.Frame(self.root,bg='#eef2f7',padx=20,pady=14)
        footer.grid(row=3,column=0,sticky='ew')
        self.progress=ttk.Progressbar(footer,mode='indeterminate')
        self.progress.pack(fill='x')
        self.status=tk.StringVar(value='Sẵn sàng. Ảnh lớn được thu nhỏ theo cạnh tối đa đã chọn, giữ nguyên tỷ lệ.')
        tk.Label(footer,textvariable=self.status,font=('Segoe UI',10),bg='#eef2f7',fg='#334155',anchor='w',justify='left',wraplength=1120).pack(fill='x',pady=(9,0))

    def _panel(self,parent,column,title,placeholder):
        frame=tk.Frame(parent,bg='white',highlightbackground='#dce3ed',highlightthickness=1,padx=14,pady=12)
        frame.grid(row=0,column=column,sticky='nsew',padx=(0,8) if column==0 else (8,0))
        frame.columnconfigure(0,weight=1)
        frame.rowconfigure(1,weight=1)
        tk.Label(frame,text=title,font=('Segoe UI',12,'bold'),bg='white',fg='#172b47',anchor='w').grid(row=0,column=0,sticky='ew',pady=(0,10))
        preview=tk.Label(frame,text=placeholder,bg='#f4f6fa',fg='#7b8798',font=('Segoe UI',11),anchor='center')
        preview.grid(row=1,column=0,sticky='nsew')
        caption=tk.StringVar(value='')
        tk.Label(frame,textvariable=caption,font=('Segoe UI',9),bg='white',fg='#64748b',anchor='w',wraplength=490).grid(row=2,column=0,sticky='ew',pady=(10,0))
        return preview,caption

    def choose_image(self):
        if self.busy:return
        selected=filedialog.askopenfilename(title='Chọn ảnh tối',filetypes=[('Ảnh','*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff'),('Tất cả tệp','*.*')])
        if selected:self.select_path(selected)

    def select_path(self,path):
        if self.busy:return
        try:
            original=read_photo(path)
            prepared=prepare_photo(original,int(self.resolution.get()))
        except (OSError,ValueError) as exc:
            messagebox.showerror('Không đọc được ảnh',str(exc),parent=self.root)
            return
        self.selected_path=Path(path).resolve()
        self._input_image=prepared
        self._output_image=None
        self.result=None
        self.last_output_path=None
        self.input_caption.set(f'{self.selected_path.name} | Gốc {original.width} × {original.height} | Xử lý {prepared.width} × {prepared.height}')
        self.output_caption.set('')
        self.output_preview.configure(image='',text='Nhấn Tăng sáng để xử lý ảnh')
        self.run_button.configure(state='normal')
        self.save_button.configure(state='disabled')
        self.folder_button.configure(state='disabled')
        self.status.set('Đã chọn ảnh. Nhấn Tăng sáng để chạy mô hình.')
        self._render_previews()

    def _resolution_changed(self,event=None):
        if self.selected_path and not self.busy:self.select_path(self.selected_path)

    def start_processing(self):
        if self.busy or self.selected_path is None:return
        self.busy=True
        self.service.max_side=int(self.resolution.get())
        for button in (self.choose_button,self.run_button,self.save_button,self.folder_button):button.configure(state='disabled')
        self.resolution.configure(state='disabled')
        self.progress.configure(mode='indeterminate',value=0)
        self.progress.start(10)
        self.status.set('Đang tăng sáng bằng mô hình đã huấn luyện. Lần đầu cần nạp checkpoint...')
        path=self.selected_path
        def worker():
            try:self._events.put(('done',self.service.enhance_file(path)))
            except Exception as exc:
                logging.exception('Image processing failed')
                self._events.put(('error',str(exc)))
        threading.Thread(target=worker,name='image-enhancement',daemon=True).start()

    def _poll(self):
        try:
            state,payload=self._events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy=False
            self.progress.stop()
            self.choose_button.configure(state='normal')
            self.run_button.configure(state='normal' if self.selected_path else 'disabled')
            self.resolution.configure(state='readonly')
            if state=='done':
                self.result=payload
                self.last_output_path=payload.output_path
                self.progress.configure(mode='determinate',value=100)
                self._input_image=payload.input_image
                self._output_image=payload.output_image
                self.output_caption.set(f'PNG RGB | {payload.processed_size[0]} × {payload.processed_size[1]} | {payload.seconds:.2f} giây')
                self.status.set(f'Đã lưu: {payload.output_path}')
                self.save_button.configure(state='normal')
                self.folder_button.configure(state='normal')
                self._render_previews()
            else:
                self.progress.configure(mode='determinate',value=0)
                self.status.set('Xử lý chưa thành công. Kiểm tra ảnh và thử lại.')
                if not self._closing:messagebox.showerror('Không xử lý được ảnh',payload,parent=self.root)
            if self._closing:
                self.root.destroy()
                return
        self._poll_job=self.root.after(100,self._poll)

    def _schedule_render(self,event=None):
        if event is not None and event.widget!=self.root:return
        if self._render_job is not None:self.root.after_cancel(self._render_job)
        self._render_job=self.root.after(150,self._render_previews)

    def _render_previews(self):
        self._render_job=None
        self._photos=[]
        for label,image in ((self.input_preview,self._input_image),(self.output_preview,self._output_image)):
            if image is None:continue
            tile=image.copy()
            tile.thumbnail((max(label.winfo_width()-12,120),max(label.winfo_height()-12,120)),Image.Resampling.LANCZOS)
            photo=ImageTk.PhotoImage(tile)
            self._photos.append(photo)
            label.configure(image=photo,text='')

    def save_copy(self):
        if self.busy or self.result is None:return
        target=filedialog.asksaveasfilename(title='Lưu ảnh kết quả',defaultextension='.png',initialfile=f'{self.selected_path.stem}_enhanced.png',filetypes=[('PNG RGB','*.png')])
        if not target:return
        try:
            path=Path(target).resolve()
            self.service._check_output(self.selected_path,path)
            save_png(self.result.output_image,path,overwrite=True)
            self.last_output_path=path
            self.status.set(f'Đã lưu bản sao: {path}')
        except (OSError,ValueError) as exc:
            messagebox.showerror('Không lưu được ảnh',str(exc),parent=self.root)

    def open_folder(self):
        if self.result is None:return
        folder=str((self.last_output_path or self.result.output_path).parent)
        if hasattr(os,'startfile'):os.startfile(folder)
        else:subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',folder])

    def close(self):
        if self.busy:
            self._closing=True
            self.status.set('Đang hoàn tất ảnh hiện tại trước khi đóng...')
            return
        if self._render_job is not None:self.root.after_cancel(self._render_job)
        if self._poll_job is not None:self.root.after_cancel(self._poll_job)
        self.root.destroy()


def launch(threads=6,initial_image=None):
    log_dir=ROOT/'outputs/logs'
    log_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=log_dir/'desktop.log',encoding='utf-8',level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    configure_display()
    root=tk.Tk()
    app=EnhancementApp(root,EnhancementService(threads=threads))
    if initial_image:root.after(50,lambda:app.select_path(initial_image))
    root.mainloop()
