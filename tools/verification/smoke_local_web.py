"""Exercise the real local HTTP upload/result flow with the approved checkpoint."""
from pathlib import Path
import io,json,re,socket,subprocess,sys,time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from kind_enhance.ui.local_web import MAX_UPLOAD
from kind_enhance.infrastructure.runtime import sha256
from kind_enhance.inference.pipeline import configuration
from kind_enhance.infrastructure.runtime import write_json


def main():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    checks=ROOT/'outputs/checks';checks.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'checkpoints/model_current.pt'
    before=sha256(checkpoint)
    log=(checks/'app_smoke_server.log').open('wb')
    process=subprocess.Popen([sys.executable,'-B',str(ROOT/'app.py'),'--port',str(port),'--no-browser'],
                             cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
    base=f'http://127.0.0.1:{port}'
    def request(path,data=None,token=None,headers=None):
        values={'X-Local-Token':token} if token else {}
        values.update(headers or {})
        with urlopen(Request(base+path,data=data,headers=values),timeout=120) as reply:return reply.read()
    def expect_status(code,path,data=None,token=None,headers=None):
        try:request(path,data,token,headers)
        except HTTPError as exc:assert exc.code==code,(exc.code,code);return
        raise AssertionError(f'Expected {code}: {path}')
    try:
        deadline=time.monotonic()+30
        while True:
            if process.poll() is not None:raise AssertionError('app.py exited; see app_smoke_server.log')
            try:
                if json.loads(request('/api/health'))['ready']:break
            except OSError:
                if time.monotonic()>=deadline:raise AssertionError('app.py startup timeout')
                time.sleep(.1)
        page=request('/').decode('utf-8')
        token=re.search(r'name="local-token" content="([^"]+)"',page)[1]
        assert 'webkitdirectory' in page and 'Chọn thư mục' in page and 'Chọn ảnh' in page
        assert request('/app.js') and request('/app.css')
        assert json.loads(request('/api/health'))['ready']
        expect_status(403,'/api/enhance',b'no-token')
        expect_status(400,'/api/enhance',b'not-an-image',token)
        expect_status(413,'/api/enhance',b'x',token,{'Content-Length':str(MAX_UPLOAD+1)})
        expect_status(404,'/../configs/model.json')
        expect_status(403,'/api/health',headers={'Host':'example.com'})
        _,manifest=configuration()
        results=[]
        for pair in manifest['splits']['test'][:2]:
            source=Path(manifest['source']['data_root'])/pair['low']
            result=json.loads(request('/api/enhance',source.read_bytes(),token,{'X-File-Name':source.name}))
            with Image.open(io.BytesIO(request(result['output']))) as actual,Image.open(ROOT/'outputs/test_100/images'/f"{pair['id']}_RGB.png") as expected:
                assert actual.mode=='RGB' and np.array_equal(np.asarray(actual),np.asarray(expected))
            with Image.open(io.BytesIO(request(result['input']))) as preview:
                assert preview.mode=='RGB' and preview.size==tuple(result['size'])
            results.append(dict(id=pair['id'],RGB=True,pixel_identical_to_approved=True,seconds=result['seconds']))
        assert results[0]['id']!=results[1]['id']
        assert sha256(checkpoint)==before
        proof=dict(verified=True,entrypoint='app.py',launched_as_subprocess=True,started_outside_project_directory=True,
                   checkpoint_sha256=before,real_HTTP_upload=True,multiple_images_verified=True,file_and_folder_controls_present=True,
                   input_and_output_previews_verified=True,invalid_upload_rejected=True,local_access_checks_passed=True,
                   approved_checkpoint_unchanged=True,training_performed=False,results=results)
        write_json(ROOT/'outputs/checks/local_web_smoke.json',proof)
        print(json.dumps(proof),flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        log.close()


if __name__=='__main__':main()
