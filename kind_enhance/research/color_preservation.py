"""Parameter-free RGB colour preservation at fixed output luminance."""
import torch
from torch.nn import functional as F
from kind_enhance.inference.calibration import luma

def preserve_color(prediction,low,blend=.75,gain=.75):
 if not 0<=blend<=1 or not 0<=gain<=1.5:raise ValueError('Invalid colour parameters')
 if blend==0:return prediction
 y=luma(prediction)
 smooth=F.avg_pool2d(low,3,1,1,count_include_pad=False);source_y=luma(smooth)
 source_chroma=(smooth-source_y)/(source_y+.02)*y*gain
 c=(1-blend)*(prediction-y)+blend*source_chroma
 c=c-luma(c)
 # Scale chroma into the RGB gamut without changing luminance or hue direction.
 pos=torch.where(c>1e-8,(1-y)/c.clamp_min(1e-8),torch.ones_like(c))
 neg=torch.where(c< -1e-8,y/(-c).clamp_min(1e-8),torch.ones_like(c))
 scale=torch.minimum(pos,neg).amin(1,keepdim=True).clamp(0,1)*.999
 return (y+scale*c).clamp(0,1)

def enhance_rgb(model,low,config):
 from kind_enhance.inference.pipeline import enhance_calibrated
 prediction=enhance_calibrated(model,low,config)
 settings=config.get('rgb_preservation')
 return preserve_color(prediction,low,**settings) if settings else prediction
