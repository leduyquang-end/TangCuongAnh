"""GT-supported edge refinement of the existing reflectance restoration branch."""
import torch
from torch.nn import functional as F
from kind_enhance.inference.calibration import luma,calibrate
from kind_enhance.training.losses.contrast_gradient import stable_calibration_features,soft_entropy
from kind_enhance.inference.color_preservation import preserve_color
from kind_enhance.inference.luminance import refine_luminance
from kind_enhance.evaluation.reference import ssim

def stable_edges(gray):
 origin=gray[...,:-1,:-1];dx=gray[...,:-1,1:]-origin;dy=gray[...,1:,:-1]-origin
 return ((dx.square()+dy.square())/2+1e-8).sqrt()-1e-4

def supported_edge_terms(prediction,target,reference,gain):
 target,reference=target.detach(),reference.detach();edge_loss=prediction.new_zeros(());flat_loss=prediction.new_zeros(())
 for scale in (1,2):
  p,t,r=[F.avg_pool2d(x,scale) for x in (prediction,target,reference)]
  for axis in (-1,-2):
   actual,truth,prior=[x.diff(dim=axis) for x in (p,t,r)]
   support=(truth.abs()/.025).clamp(0,1).detach()
   goal=(gain*truth).clamp(-.25,.25).detach()
   weighted=((actual-goal).abs()*support).sum((-2,-1))/support.sum((-2,-1)).clamp_min(1)
   edge_loss=edge_loss+weighted.mean()
   flat_loss=flat_loss+(F.relu(actual.abs()-prior.abs())*(1-support)).mean()
 support=(stable_edges(target)/.025).clamp(0,1).detach()
 actual=stable_edges(prediction);prior=stable_edges(reference)
 has_edges=(support.mean((-2,-1))>.01).detach()
 goal=(prior*support).mean((-2,-1))*gain
 deficit=(F.relu(goal-(actual*support).mean((-2,-1)))*has_edges).mean()
 return {'signed_edges':edge_loss/4,'flat_noise':flat_loss/4,'supported_ag_deficit':deficit}

def student_output(model,low,config):
 raw=model(low)
 calibrated=calibrate(raw,stable_calibration_features(low,raw),config['calibration'])
 colored=preserve_color(calibrated,low,**config['rgb_preservation'])
 return refine_luminance(colored,**config['luminance_refinement'])

def make_ag_loss(teacher,config):
 from kind_enhance.inference.luminance import enhance_all_gain
 def objective(model,low,high,recipe):
  prediction=student_output(model,low,config)
  with torch.no_grad():reference=enhance_all_gain(teacher,low,config)
  p,t,r=[luma(x) for x in (prediction,high,reference)]
  edges=supported_edge_terms(p,t,r,recipe['edge_gain'])
  pm,rm=p.mean((-2,-1)),r.mean((-2,-1))
  ps=(p.var((-2,-1),unbiased=False)+1e-8).sqrt();rs=(r.var((-2,-1),unbiased=False)+1e-8).sqrt()
  ci=F.relu(rs*recipe['ci_gain']-ps).mean()+F.relu(ps-rs*1.12).mean()
  brightness=(F.relu(rm-pm)+F.relu(pm-1.01*rm)).mean()
  pe,re=soft_entropy(p),soft_entropy(r)
  entropy=(F.relu(re-pe)+F.relu(pe-1.01*re)).mean()
  loss=(F.l1_loss(prediction,high)+.25*(1-ssim(prediction,high)).mean()
   +recipe['edge_weight']*edges['signed_edges']+recipe['ag_weight']*edges['supported_ag_deficit']
   +recipe['noise_weight']*edges['flat_noise']+recipe['ci_weight']*ci
   +recipe['luma_anchor']*F.l1_loss(p,r)+12*brightness+.5*entropy)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite AG branch loss')
  return loss
 return objective

