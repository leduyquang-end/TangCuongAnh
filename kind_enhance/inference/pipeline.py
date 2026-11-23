import statistics
import time
from pathlib import Path
import numpy as np
import torch
from kind_enhance.models import KinDAttentionCurve
from kind_enhance.data.paired_dataset import PairedLowLightDataset,read_rgb
from kind_enhance.inference.calibration import calibrate,calibration_features,luma
from kind_enhance.evaluation.assessment import aggregate,describe
from kind_enhance.infrastructure.artifacts import save_image,write_csv,STAGES,contact_sheet
from kind_enhance.evaluation.competition import METRIC_NAMES,PDF_BENCHMARK
from kind_enhance.infrastructure.runtime import read_checkpoint,sha256,write_json,image_on_device,resource_snapshot
import json


ROOT=Path(__file__).resolve().parents[2]


def configuration():
    config=json.loads((ROOT/'configs/model.json').read_text(encoding='utf-8'))
    manifest_path=ROOT/config['manifest']
    if sha256(manifest_path)!=config['manifest_sha256']:raise ValueError('Dataset split changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    # Resolve a cloned checkout without changing the hash-protected manifest on disk.
    local_data=ROOT/'LOL-v2/LOL-v2/Real_captured'
    if (local_data/'Train/Low').is_dir() and (local_data/'Test/Low').is_dir():
        manifest['source']['data_root']=str(local_data.resolve())
    return config,manifest


def load_inference_model(key='model'):
    config=json.loads((ROOT/'configs/model.json').read_text(encoding='utf-8'))
    path=ROOT/config[key]
    if sha256(path)!=config[key+'_sha256']:
        raise ValueError(f'Checkpoint changed: {path}')
    state=read_checkpoint(path,torch.device('cpu'),config['manifest_sha256'],'enhance')
    model=KinDAttentionCurve().to(memory_format=torch.channels_last)
    model.load_state_dict(state['model'],strict=True)
    return model.eval(),config


def load_model(key='model'):
    config,manifest=configuration()
    model,loaded_config=load_inference_model(key)
    if loaded_config!=config:
        raise ValueError('Configuration changed while loading the model')
    return model,config,manifest


def enhance_calibrated(model,low,config):
    raw=model(low)
    return calibrate(raw,calibration_features(low,raw),config['calibration'])


def enhance(model,low,config):
    from kind_enhance.inference.color_preservation import preserve_color
    from kind_enhance.inference.luminance import refine_luminance
    output=enhance_calibrated(model,low,config)
    if config.get('rgb_preservation'):
        output=preserve_color(output,low,**config['rgb_preservation'])
    if config.get('luminance_refinement'):
        output=refine_luminance(output,**config['luminance_refinement'])
    return output


def infer(args):
    from kind_enhance.application.enhancement import EnhancementService
    service=EnhancementService(threads=args.threads,max_side=getattr(args,'max_side',0))
    result=service.enhance_file(args.input,args.output,overwrite=getattr(args,'overwrite',False),stages=args.stages)
    print(json.dumps(result.metadata(),ensure_ascii=False,indent=2))


@torch.inference_mode()
def measure(model,dataset,config,collect=False):
    model.eval()
    rows=[]
    for low,high,image_id in dataset:
        low,high=[image_on_device(x.unsqueeze(0),'cpu') for x in (low,high)]
        pred=enhance(model,low,config)
        rows.append({'id':image_id,**describe(pred,high),
                     'chroma_mae':torch.nn.functional.l1_loss(pred-luma(pred),high-luma(high)).item()})
    result={**aggregate(rows),'chroma_mae':statistics.mean(r['chroma_mae'] for r in rows)}
    return (result,rows) if collect else result


@torch.inference_mode()
def evaluate(args):
    model,config,manifest=load_model()
    split=manifest['splits'][args.split]
    if args.split=='test' and (len(split)!=100 or {r['id'] for r in split}!={f'{i:05d}' for i in range(690,790)}):
        raise ValueError('Expected the unchanged official 100-pair test split')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'evaluation_lock.json',{'checkpoint_sha256':config['model_sha256'],
        'calibration':config['calibration'],'rgb_preservation':config.get('rgb_preservation'),
        'luminance_refinement':config.get('luminance_refinement'),'manifest_sha256':config['manifest_sha256'],'split':args.split})
    rows=[]
    start=time.perf_counter()
    for index,(low,high,image_id) in enumerate(PairedLowLightDataset(manifest['source']['data_root'],split),1):
        low,high=[image_on_device(x.unsqueeze(0),'cpu') for x in (low,high)]
        pred=enhance(model,low,config)
        rows.append({'id':image_id,**describe(pred,high)})
        if args.save_images:
            images=args.output/'images'
            images.mkdir(exist_ok=True)
            save_image(images/f'{image_id}_output.png',pred)
        if index%25==0:print(f'{args.split}: {index}/{len(split)}',flush=True)
    mean=aggregate(rows)
    std={key:statistics.stdev(r[key] for r in rows) for key in (*METRIC_NAMES,'PSNR','SSIM')}
    write_csv(args.output/'per_image.csv',rows)
    write_json(args.output/'metrics.json',{'mean':mean,'sample_std':std,'seconds':time.perf_counter()-start,
        'resources':resource_snapshot(torch.device('cpu'))})
    print(json.dumps(mean,indent=2))
