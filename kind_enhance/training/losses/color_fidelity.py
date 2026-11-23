"""Colour fidelity with explicit anti-desaturation supervision from paired GT."""
import torch
from torch.nn import functional as F
from kind_enhance.training.losses.color import chroma,chromaticity,color_terms
from kind_enhance.inference.calibration import luma,calibrate
from kind_enhance.training.losses.base import refinement_terms
from kind_enhance.training.losses.contrast_gradient import stable_calibration_features,soft_entropy

def fidelity_terms(prediction,target):
 target=target.detach()
 p,t=chroma(prediction),chroma(target)
 def sat(x):return (x.amax(1,keepdim=True)-x.amin(1,keepdim=True))/(x.amax(1,keepdim=True)+.02)
 saturation=(sat(prediction)-sat(target)).abs().mean()
 spread=((prediction.amax(1)-prediction.amin(1))-(target.amax(1)-target.amin(1))).abs().mean()
 direction=((F.normalize(p,dim=1,eps=.01)-F.normalize(t,dim=1,eps=.01)).abs()*sat(target)).mean()
 return {'saturation':saturation,'colour_spread':spread,'direction':direction}

def make_color_loss(teacher,output_config):
 from kind_enhance.inference.pipeline import enhance
 def objective(model,low,high,recipe):
  terms,raw=refinement_terms(model,low,high,recipe,return_prediction=True)
  final=calibrate(raw,stable_calibration_features(low,raw),output_config['calibration'])
  with torch.no_grad():reference=enhance(teacher,low,output_config)
  fc,rc=color_terms(final,high),color_terms(raw,high)
  fidelity=fidelity_terms(final,high)
  loss=sum(terms.values())+recipe['color_weight']*(fc['chroma']+.5*rc['chroma'])+recipe['ratio_weight']*fc['chromaticity']
  for key in ('saturation','colour_spread','direction'):loss=loss+recipe[key+'_weight']*fidelity[key]
  low_r,_=model.decomposition(low)
  loss=loss+recipe['decomposition_color_weight']*color_terms(low_r,high)['chromaticity']
  actual,target=luma(final),luma(reference)
  loss=loss+recipe['luma_teacher_weight']*F.l1_loss(actual,target)
  loss=loss+12*F.relu((actual.mean((-2,-1))-target.mean((-2,-1))).abs()-.005*target.mean((-2,-1))).mean()
  loss=loss+.25*(soft_entropy(actual)-soft_entropy(target)).abs().mean()
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite colour fidelity loss')
  return loss
 return objective
