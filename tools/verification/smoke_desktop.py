"""Exercise the desktop workflow with the real checkpoint and capture its window."""
from pathlib import Path
import json,shutil,sys,time,tkinter as tk
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageGrab
from kind_enhance.application.enhancement import EnhancementService
from kind_enhance.ui.desktop import EnhancementApp,configure_display
from kind_enhance.infrastructure.runtime import write_json


def main():
    cfg=json.loads((ROOT/'configs/model.json').read_text())
    manifest=json.loads((ROOT/cfg['manifest']).read_text())
    source=Path(manifest['source']['data_root'])/manifest['splits']['test'][0]['low']
    demo=ROOT/'assets/demo';demo.mkdir(parents=True,exist_ok=True)
    input_path=demo/'anh_toi.png';shutil.copyfile(source,input_path)
    configure_display()
    root=tk.Tk();root.geometry('1180x760+40+40')
    app=EnhancementApp(root,EnhancementService(threads=6,max_side=1280))
    errors=[]
    try:
        with patch('kind_enhance.ui.desktop.messagebox.showerror',side_effect=lambda title,message,**kwargs:errors.append(message)):
            root.update()
            with patch('kind_enhance.ui.desktop.filedialog.askopenfilename',return_value=str(input_path)):
                app.choose_image()
            root.update();app.start_processing()
            deadline=time.monotonic()+90
            while app.busy and time.monotonic()<deadline:
                root.update();time.sleep(.02)
            assert not app.busy and not errors and app.result is not None,errors
            app._render_previews();root.update()
            until=time.monotonic()+.5
            while time.monotonic()<until:root.update();time.sleep(.02)
            expected=ROOT/'experiments/ag_refinement/run_four_metric_curriculum/test/images/00690_RGB.png'
            with Image.open(expected) as old:
                assert np.array_equal(np.asarray(app.result.output_image),np.asarray(old))
            output=demo/'anh_sang.png'
            with patch('kind_enhance.ui.desktop.filedialog.asksaveasfilename',return_value=str(output)):
                app.save_copy()
            with Image.open(output) as copied:assert np.array_equal(np.asarray(copied),np.asarray(app.result.output_image))
            if sys.platform=='win32':
                with patch('kind_enhance.ui.desktop.os.startfile') as open_folder:
                    app.open_folder();open_folder.assert_called_once_with(str(demo))
            screenshots=ROOT/'doc/3_san_pham/assets';screenshots.mkdir(parents=True,exist_ok=True)
            root.lift();root.attributes('-topmost',True);root.update()
            time.sleep(.2)
            box=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())
            screenshot=screenshots/'giao_dien_ung_dung.png';ImageGrab.grab(bbox=box).save(screenshot)
            report=dict(verified=True,workflow=['select_photo','background_inference','RGB_preview','automatic_PNG_save','save_copy','open_latest_output_folder'],
                        output_RGB=True,PNG_identical_to_approved=True,input_image=str(input_path),output_image=str(output),
                        screenshot=str(screenshot),processing=app.result.metadata(),errors=errors)
            checks=ROOT/'outputs/checks';checks.mkdir(parents=True,exist_ok=True);write_json(checks/'desktop_smoke.json',report)
            print(json.dumps(report),flush=True)
    finally:
        app.close()


if __name__=='__main__':main()
