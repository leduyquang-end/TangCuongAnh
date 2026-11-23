"""Small localhost interface using the same approved enhancement service."""
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import unquote, urlsplit
import uuid
import webbrowser

from kind_enhance.application.enhancement import EnhancementService
from kind_enhance.inference.pipeline import ROOT
from kind_enhance.infrastructure.images import save_png

ASSETS=Path(__file__).with_name('web')
MAX_UPLOAD=20*1024*1024
LOGGER=logging.getLogger(__name__)


class LocalServer(ThreadingHTTPServer):
    daemon_threads=True

    def __init__(self,port=7860,threads=6):
        super().__init__(('127.0.0.1',port),LocalHandler)
        self.token=secrets.token_urlsafe(32)
        self.service=EnhancementService(threads=threads,max_side=1280)
        self.results={}
        self.results_lock=threading.Lock()
        self.uploads=ROOT/'outputs/uploads'
        self.uploads.mkdir(parents=True,exist_ok=True)
        self.url=f'http://localhost:{self.server_port}'


class LocalHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(60)

    def log_message(self,fmt,*args):
        LOGGER.info(fmt,*args)

    def send_data(self,status,body,content_type):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

    def send_json(self,status,data):
        self.send_data(status,json.dumps(data,ensure_ascii=False).encode('utf-8'),'application/json; charset=utf-8')

    def valid_host(self):
        return self.headers.get('Host') in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}')

    def do_GET(self):
        if not self.valid_host():
            self.send_json(403,{'error':'Địa chỉ truy cập không hợp lệ.'});return
        path=urlsplit(self.path).path
        if path=='/':
            page=(ASSETS/'index.html').read_text(encoding='utf-8').replace('__LOCAL_TOKEN__',self.server.token)
            self.send_data(200,page.encode('utf-8'),'text/html; charset=utf-8')
        elif path in ('/app.css','/app.js'):
            file=ASSETS/path[1:]
            mime='text/css' if path.endswith('.css') else 'text/javascript'
            self.send_data(200,file.read_bytes(),mime+'; charset=utf-8')
        elif path=='/api/health':
            self.send_json(200,{'app':'kind-enhance-local','ready':True})
        elif match:=re.fullmatch(r'/images/([0-9a-f]{32})/(input|output)\.png',path):
            with self.server.results_lock:
                files=self.server.results.get(match[1])
            if not files:
                self.send_json(404,{'error':'Không tìm thấy ảnh.'});return
            self.send_data(200,files[match[2]].read_bytes(),'image/png')
        elif path=='/favicon.ico':
            self.send_data(204,b'','image/x-icon')
        else:
            self.send_json(404,{'error':'Không tìm thấy trang.'})

    def do_POST(self):
        if not self.valid_host() or not secrets.compare_digest(self.headers.get('X-Local-Token',''),self.server.token):
            self.send_json(403,{'error':'Hãy tải lại trang ứng dụng rồi thử lại.'});return
        if urlsplit(self.path).path!='/api/enhance':
            self.send_json(404,{'error':'Không tìm thấy chức năng.'});return
        try:length=int(self.headers.get('Content-Length','0'))
        except ValueError:length=0
        if not 0<length<=MAX_UPLOAD:
            self.send_json(413,{'error':'Chọn ảnh có dung lượng từ 1 byte đến 20 MB.'});return
        image_id=uuid.uuid4().hex
        source=self.server.uploads/f'{image_id}.upload'
        preview=self.server.uploads/f'{image_id}_preview.png'
        try:
            payload=self.rfile.read(length)
            if len(payload)!=length:raise ValueError('Ảnh tải lên chưa đầy đủ.')
            source.write_bytes(payload)
            name=unquote(self.headers.get('X-File-Name','anh_toi')).replace('\\','/').rsplit('/',1)[-1]
            stem=''.join(c for c in Path(name).stem if c.isalnum() or c in '-_ ')[:70].strip() or 'anh'
            output=ROOT/'outputs/images'/f'{stem}_{image_id[:8]}.png'
            result=self.server.service.enhance_file(source,output)
            save_png(result.input_image,preview)
            with self.server.results_lock:
                self.server.results[image_id]={'input':preview,'output':output}
            self.send_json(200,{'input':f'/images/{image_id}/input.png','output':f'/images/{image_id}/output.png',
                                'filename':output.name,'size':list(result.processed_size),'seconds':round(result.seconds,3)})
        except (ValueError,OSError) as exc:
            source.unlink(missing_ok=True)
            self.send_json(400,{'error':str(exc)})
        except Exception:
            LOGGER.exception('Photo processing failed')
            self.send_json(500,{'error':'Không xử lý được ảnh. Kiểm tra log hoặc thử ảnh khác.'})


def launch(port=7860,threads=6,open_browser=True):
    if not 1<=port<=65535:raise ValueError('Cổng phải nằm trong khoảng 1–65535.')
    logs=ROOT/'outputs/logs'
    logs.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=logs/'web.log',encoding='utf-8',level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    try:server=LocalServer(port,threads)
    except OSError:
        from urllib.request import urlopen
        try:
            with urlopen(f'http://127.0.0.1:{port}/api/health',timeout=2) as response:
                existing=json.load(response)
            if existing.get('app')!='kind-enhance-local':raise ValueError('Cổng đang được ứng dụng khác sử dụng.')
        except (OSError,ValueError) as exc:
            raise ValueError(f'Không mở được cổng {port}. Dùng --port để chọn cổng khác.') from exc
        if open_browser:webbrowser.open(f'http://localhost:{port}')
        print(f'Ứng dụng đã chạy tại http://localhost:{port}',flush=True)
        return
    print(f'Mở giao diện: {server.url}\nGiữ cửa sổ này khi dùng ứng dụng. Nhấn Ctrl+C để dừng.',flush=True)
    if open_browser:threading.Timer(.4,lambda:webbrowser.open(server.url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
