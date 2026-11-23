"""Run the approved learned pipeline for a user photo; no dataset or GT is required."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import logging
import threading
import time
import uuid
import numpy as np
import torch
from PIL import Image
from kind_enhance.inference.pipeline import ROOT,enhance,load_inference_model
from kind_enhance.infrastructure.artifacts import pixels,save_image,STAGES
from ..infrastructure.images import read_photo,prepare_photo,save_png
from kind_enhance.infrastructure.runtime import write_json

LOGGER=logging.getLogger(__name__)


@dataclass
class EnhancementResult:
    input_path:Path
    output_path:Path
    original_size:tuple
    processed_size:tuple
    seconds:float
    checkpoint_sha256:str
    input_image:Image.Image
    output_image:Image.Image

    def metadata(self):
        return dict(input_path=str(self.input_path),output_path=str(self.output_path),
                    original_size=list(self.original_size),processed_size=list(self.processed_size),
                    seconds=self.seconds,checkpoint_sha256=self.checkpoint_sha256,
                    output_mode='RGB',training_performed=False)


class EnhancementService:
    def __init__(self,threads=6,max_side=1280):
        if threads<1:raise ValueError('Số luồng phải lớn hơn 0.')
        if max_side!=0 and max_side<32:raise ValueError('Cạnh tối đa phải >=32, hoặc 0.')
        self.threads,self.max_side=threads,max_side
        self._model=None;self._config=None;self._lock=threading.Lock()

    def _load(self):
        if self._model is None:
            torch.set_num_threads(self.threads)
            self._model,self._config=load_inference_model()
            LOGGER.info('Loaded approved checkpoint %s',self._config['model_sha256'])

    @staticmethod
    def _check_output(input_path,output_path):
        if output_path==input_path:
            raise ValueError('Không ghi đè ảnh gốc. Hãy chọn tên ảnh kết quả khác.')
        for directory in ('LOL-v2','data','checkpoints','configs','kind_enhance'):
            if output_path.is_relative_to((ROOT/directory).resolve()):
                raise ValueError('Hãy lưu ảnh kết quả ngoài thư mục dữ liệu/mã nguồn của dự án.')

    @torch.inference_mode()
    def enhance_file(self,input_path,output_path=None,overwrite=False,stages=None):
        started=time.perf_counter()
        input_path=Path(input_path).expanduser().resolve()
        source=read_photo(input_path)
        image=prepare_photo(source,self.max_side)
        if output_path is None:
            name=f'{input_path.stem}_enhanced_{uuid.uuid4().hex[:8]}.png'
            output_path=ROOT/'outputs/images'/name
        output_path=Path(output_path).expanduser().resolve()
        self._check_output(input_path,output_path)
        if output_path.suffix.lower()!='.png':raise ValueError('Kết quả được lưu dạng PNG RGB (.png).')
        if output_path.exists() and not overwrite:raise FileExistsError(f'Tệp đã tồn tại: {output_path.name}')
        stages=Path(stages).expanduser().resolve() if stages is not None else None
        if stages is not None:
            if stages.exists():raise FileExistsError('Thư mục ảnh thành phần đã tồn tại; hãy chọn thư mục mới.')
            self._check_output(input_path,stages/'component.png')
        with self._lock:
            self._load()
            array=np.asarray(image,dtype=np.uint8).copy()
            tensor=torch.from_numpy(array).permute(2,0,1).float().div_(255)[None].contiguous(memory_format=torch.channels_last)
            output=enhance(self._model,tensor,self._config)
            if output.shape!=tensor.shape or not torch.isfinite(output).all():
                raise RuntimeError('Mô hình không tạo được ảnh hợp lệ.')
            result_image=Image.fromarray(pixels(output))
            save_png(result_image,output_path,overwrite)
            if stages is not None:
                stages.mkdir(parents=True,exist_ok=False)
                r,l=self._model.decomposition(tensor)
                raw,restored,adjusted,_=self._model.enhance(r,l)
                for (name,_),value in zip(STAGES[:7],(tensor,r,l,r*l,restored,adjusted,raw)):
                    save_image(stages/f'{name}.png',value)
                save_image(stages/'08_final_RGB.png',output)
            result=EnhancementResult(input_path,output_path,source.size,image.size,time.perf_counter()-started,
                                     self._config['model_sha256'],image,result_image)
        run_dir=ROOT/'outputs/runs'
        try:
            run_dir.mkdir(parents=True,exist_ok=True)
            record_key=hashlib.sha256(str(output_path).casefold().encode('utf-8')).hexdigest()[:12]
            write_json(run_dir/f'{output_path.stem}_{record_key}.json',result.metadata())
        except OSError:
            LOGGER.warning('Could not save optional run metadata',exc_info=True)
        LOGGER.info('Saved image %s in %.3fs',output_path,result.seconds)
        return result
