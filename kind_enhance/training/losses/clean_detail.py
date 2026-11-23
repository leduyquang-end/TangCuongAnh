"""Paired detail supervision that rewards correctly directed edges and penalizes flat-region noise."""
import torch
from torch.nn import functional as F
from kind_enhance.training.losses.supported_edges import stable_edges,student_output
from kind_enhance.inference.pipeline import enhance
from kind_enhance.training.losses.contrast_gradient import soft_entropy
from kind_enhance.evaluation.reference import ssim
from kind_enhance.inference.calibration import luma


def differences(gray):
    origin=gray[...,:-1,:-1]
    return gray[...,:-1,1:]-origin,gray[...,1:,:-1]-origin


def clean_structure_terms(prediction,target,reference,gain):
    target,reference=target.detach(),reference.detach()
    px,py=differences(prediction)
    tx,ty=differences(target)
    rx,ry=differences(reference)
    magnitude=(tx.square()+ty.square()+1e-8).sqrt()
    support=((magnitude-.004)/.025).clamp(0,1).detach()
    ux,uy=tx/magnitude,ty/magnitude
    projection=px*ux+py*uy
    prior_projection=(rx*ux+ry*uy).clamp_min(0)
    # Positive projection rewards the correct GT orientation, not arbitrary edge magnitude.
    wanted=prior_projection*gain
    deficit=(F.relu(wanted-projection)*support).sum()/support.sum().clamp_min(1)
    perpendicular=(px*uy-py*ux).abs()
    wrong_direction=(F.relu(-projection)*support).sum()/support.sum().clamp_min(1)
    off_direction=(perpendicular*support).sum()/support.sum().clamp_min(1)
    local_strength=F.avg_pool2d(magnitude,3,1,1,count_include_pad=False)
    flat=(1-local_strength/.008).clamp(0,1).detach()
    flat_gradient=(((px-tx).abs()+(py-ty).abs())*flat).sum()/(2*flat.sum().clamp_min(1))
    def highpass(x):return x-F.avg_pool2d(x,3,1,1,count_include_pad=False)
    padded_flat=F.pad(flat,(0,1,0,1),mode='replicate')
    flat_texture=((highpass(prediction)-highpass(target)).abs()*padded_flat).sum()/padded_flat.sum().clamp_min(1)
    return dict(aligned_deficit=deficit,wrong_direction=wrong_direction,
                perpendicular=off_direction,flat_gradient=flat_gradient,flat_texture=flat_texture)


@torch.no_grad()
def guided_detail_target(target,reference,detail_gain):
    target,reference=target.detach(),reference.detach()
    mean=target.mean((-2,-1),keepdim=True)
    spread=(target.var((-2,-1),keepdim=True,unbiased=False)+1e-8).sqrt()
    ref_mean=reference.mean((-2,-1),keepdim=True)
    ref_spread=(reference.var((-2,-1),keepdim=True,unbiased=False)+1e-8).sqrt()
    # Do not recreate the teacher's noise variance when a clean GT crop is flat.
    ratio=(ref_spread/spread).clamp(.5,2.)
    matched=(target-mean)*ratio+ref_mean
    dx,dy=differences(target)
    norm=(dx.square()+dy.square()+1e-8).sqrt()
    support=F.pad(((norm-.004)/.025).clamp(0,1),(0,1,0,1),mode='replicate')
    detail=matched-F.avg_pool2d(matched,3,1,1,count_include_pad=False)
    boosted=matched+detail_gain*detail.clamp(-.12,.12)*support
    boosted_mean=boosted.mean((-2,-1),keepdim=True)
    boosted_spread=(boosted.var((-2,-1),keepdim=True,unbiased=False)+1e-8).sqrt()
    # Keep global contrast a structural goal, without manufacturing variation in flat crops.
    desired_spread=torch.minimum(ref_spread*1.005,spread*2.0)
    return ((boosted-boosted_mean)*desired_spread/boosted_spread+ref_mean).clamp(0,1)


def make_clean_detail_loss(teacher,output_config):
    def objective(model,low,high,recipe):
        prediction=student_output(model,low,output_config)
        with torch.no_grad():reference=enhance(teacher,low,output_config)
        actual,truth,prior=[luma(x) for x in (prediction,high,reference)]
        goal=guided_detail_target(truth,prior,recipe['detail_gain'])
        terms=clean_structure_terms(actual,truth,prior,recipe['aligned_gain'])
        signed=actual.new_zeros(())
        for scale in (1,2):
            a,g=[F.avg_pool2d(x,scale) for x in (actual,goal)]
            for axis in (-1,-2):signed=signed+F.l1_loss(a.diff(dim=axis),g.diff(dim=axis))
        mean,ref_mean=actual.mean((-2,-1)),prior.mean((-2,-1))
        brightness=(F.relu(ref_mean*1.002-mean)+F.relu(mean-ref_mean*1.015)).mean()
        spread=lambda x:(x.var((-2,-1),unbiased=False)+1e-8).sqrt()
        textured=(spread(truth)>.02).float().detach()
        contrast=((F.relu(spread(prior)-spread(actual))+F.relu(spread(actual)-1.03*spread(prior)))*textured).mean()
        entropy,ref_entropy=soft_entropy(actual,bins=64),soft_entropy(prior,bins=64)
        entropy_floor=(F.relu(ref_entropy-entropy)*textured.flatten(1).mean(1)).mean()
        low_frequency=F.l1_loss(F.avg_pool2d(actual,8),F.avg_pool2d(prior,8))
        loss=(3*F.l1_loss(actual,goal)+2*(1-ssim(prediction,high)).mean()
              +recipe['signed_weight']*signed/4
              +recipe['projection_weight']*(terms['aligned_deficit']+terms['wrong_direction']+terms['perpendicular'])
              +recipe['noise_weight']*(terms['flat_gradient']+terms['flat_texture'])
              +80*brightness+20*contrast+4*entropy_floor+.3*low_frequency)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite clean-detail loss')
        return loss
    return objective
