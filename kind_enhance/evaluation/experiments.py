"""Shared colour/edge measurements and checkpoint helpers for the active model."""
import hashlib
import statistics
import torch
from kind_enhance.evaluation.assessment import aggregate,describe
from kind_enhance.training.losses.color_fidelity import color_terms
from kind_enhance.inference.pipeline import load_model
from kind_enhance.infrastructure.runtime import read_checkpoint
from kind_enhance.training.losses.supported_edges import stable_edges
from kind_enhance.inference.luminance import enhance_all_gain
from kind_enhance.inference.calibration import luma


def sat(x):return (x.amax(1)-x.amin(1))/x.amax(1).clamp_min(1e-8)

def describe_color(pred,gt,name):
 return {'id':name,**describe(pred,gt),'chroma_mae':color_terms(pred,gt)['chroma'].item(),
 'chromaticity_mae':color_terms(pred,gt)['chromaticity'].item(),'saturation':sat(pred).mean().item(),
 'gt_saturation':sat(gt).mean().item(),'saturation_mae':(sat(pred)-sat(gt)).abs().mean().item()}

def means(rows):
 m=aggregate(rows)
 for k in ('chroma_mae','chromaticity_mae','saturation','gt_saturation','saturation_mae'):m[k]=statistics.mean(r[k] for r in rows)
 return m

@torch.inference_mode()
def measure(model,data,cfg):
 model.eval();rows=[]
 for low,high,name in data:
  low,high=[x[None].contiguous(memory_format=torch.channels_last) for x in (low,high)]
  prediction=enhance_all_gain(model,low,cfg);p,t=luma(prediction),luma(high)
  flat=(stable_edges(t)<.003).float();flat_ag=(stable_edges(p)*flat).sum()/flat.sum().clamp_min(1)
  rows.append({**describe_color(prediction,high,name),'flat_ag':flat_ag.item()})
 m=means(rows);m['flat_ag']=statistics.mean(r['flat_ag'] for r in rows)
 return m,rows

def failures(m,b):
 checks={k:m[k]>=b[k] for k in ('MLI','E')}
 checks.update(CI=m['CI']>=b['CI']*1.01,AG=m['AG']>=b['AG']*1.05,
  brightness_stability=m['MLI']<=b['MLI']*1.02,entropy_stability=m['E']<=b['E']*1.01,
  color=m['chroma_mae']<=b['chroma_mae']*1.02,flat_noise=m['flat_ag']<=max(b['flat_ag']*1.05,1e-5),
  PSNR=m['PSNR']>=b['PSNR']-.2,SSIM=m['SSIM']>=b['SSIM']-.005,clipping=m['clipped_fraction']<=.01)
 return [k for k,v in checks.items() if not v]

def branch_hash(model,name):
 import hashlib
 h=hashlib.sha256()
 for k,v in getattr(model,name).state_dict().items():h.update(k.encode());h.update(v.contiguous().cpu().numpy().tobytes())
 return h.hexdigest()

def load_candidate(path,cfg):
 model,_,_=load_model();model.load_state_dict(read_checkpoint(path,'cpu',cfg['manifest_sha256'],'enhance')['model']);return model
