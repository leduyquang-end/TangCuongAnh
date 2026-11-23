"""Small luminance adjustment with existing RGB chroma retained in gamut."""
import math
import torch
from .postprocessing import luma

METRICS=('MLI','CI','E','AG')
def refine_luminance(rgb,contrast=1.,lift=0.):
 if not 1<=contrast<=1.1 or not 0<=lift<=.02:raise ValueError('Invalid luminance parameters')
 if contrast==1 and lift==0:return rgb
 old_y=luma(rgb);mean=old_y.mean((-2,-1),keepdim=True)
 new_y=(mean+contrast*(old_y-mean)+lift).clamp(0,1)
 c=rgb-old_y;c=c-luma(c)
 margin=torch.minimum(new_y,1-new_y).clamp(0,1/255)
 pos=torch.where(c>1e-8,(1-margin-new_y)/c.clamp_min(1e-8),torch.ones_like(c))
 neg=torch.where(c< -1e-8,(new_y-margin)/(-c).clamp_min(1e-8),torch.ones_like(c))
 scale=torch.minimum(pos,neg).amin(1,keepdim=True).clamp(0,1)
 return (new_y+scale*c).clamp(0,1)

def all_gain_failures(metrics,baselines,margin=.001):
 failures=[]
 for k in METRICS:
  floor=max(b[k] for b in baselines)*(1+margin)
  if not math.isfinite(metrics[k]) or metrics[k]<floor:failures.append(k)
 return failures

def enhance_all_gain(model,low,config):
 from .color_preservation_v2 import enhance_rgb
 result=enhance_rgb(model,low,config)
 parameters=config.get('luminance_refinement')
 return refine_luminance(result,**parameters) if parameters else result
