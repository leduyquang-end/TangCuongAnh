"""Parameter-free RGB colour preservation at fixed output luminance."""
import torch
from torch.nn import functional as F
from .postprocessing import luma

def preserve_color(prediction,low,blend=.75,gain=.75):
 if not 0<=blend<=1 or not 0<=gain<=1.5:raise ValueError('Invalid colour parameters')
 if blend==0:return prediction
 y=luma(prediction)
 smooth=F.avg_pool2d(low,(1,15),1,(0,7),count_include_pad=False)
 smooth=F.avg_pool2d(smooth,(15,1),1,(7,0),count_include_pad=False);source_y=luma(smooth)
 source_chroma=(smooth-source_y)/(source_y+.02)*y*gain
 c=(1-blend)*(prediction-y)+blend*source_chroma
 c=c-luma(c)
 # Scale chroma into the RGB gamut without changing luminance or hue direction.
 margin=torch.minimum(y,1-y).clamp(0,1/255)
 pos=torch.where(c>1e-8,(1-margin-y)/c.clamp_min(1e-8),torch.ones_like(c))
 neg=torch.where(c< -1e-8,(y-margin)/(-c).clamp_min(1e-8),torch.ones_like(c))
 scale=torch.minimum(pos,neg).amin(1,keepdim=True).clamp(0,1)*.999
 return (y+scale*c).clamp(0,1)

def enhance_rgb(model,low,config):
 from .final_pipeline import enhance
 prediction=enhance(model,low,config)
 settings=config.get('rgb_preservation')
 return preserve_color(prediction,low,**settings) if settings else prediction

