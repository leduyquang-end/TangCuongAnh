"""Compatibility entrypoint for inference with the user-approved current pipeline."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from kind_enhance.inference.pipeline import load_model,enhance
from kind_enhance.infrastructure.runtime import sha256,read_checkpoint
from kind_enhance.data.paired_dataset import read_rgb
from kind_enhance.infrastructure.artifacts import save_image
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--selection',type=Path,default=ROOT/'results/active_model/selection.json')
a=parser.parse_args();s=json.loads(a.selection.read_text())
assert s['selection_locked'] and (s.get('user_approved') or s.get('passed_four_metric_floors') or s.get('passed_all_gates'))
assert sha256(s['checkpoint'])==s['sha256']
torch.set_num_threads(6);model,cfg,_=load_model()
model.load_state_dict(read_checkpoint(s['checkpoint'],'cpu',cfg['manifest_sha256'],'enhance')['model'])
cfg=s.get('output_config',cfg)
x=torch.from_numpy(np.ascontiguousarray(read_rgb(a.input))).permute(2,0,1).float()[None]/255
with torch.inference_mode():out=enhance(model,x.contiguous(memory_format=torch.channels_last),cfg)
a.output.parent.mkdir(parents=True,exist_ok=True);save_image(a.output,out)
print('Saved:',a.output)
