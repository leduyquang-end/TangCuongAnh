"""Training-only full-image teacher context; inference architecture stays unchanged."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from kind_enhance.data.paired_dataset import PairedLowLightDataset,read_rgb
from kind_enhance.inference.calibration import luma,calibrate,calibration_features
from kind_enhance.inference.color_preservation import preserve_color
from kind_enhance.inference.luminance import refine_luminance
from kind_enhance.training.losses.contrast_gradient import stable_calibration_features,soft_entropy
from kind_enhance.training.losses.clean_detail import clean_structure_terms,guided_detail_target,differences
from kind_enhance.training.losses.color_fidelity import color_terms
from kind_enhance.evaluation.reference import ssim
from kind_enhance.training.losses.supported_edges import stable_edges


class FullImageTeacherDataset(PairedLowLightDataset):
    def __init__(self,root,rows,teacher,config,patch=80):
        super().__init__(root,rows,patch=patch,cache=True)
        self.reference={};self.context={};started=time.perf_counter()
        with torch.inference_mode():
            for index,row in enumerate(rows):
                low,high=[read_rgb(self.root/row[key]) for key in ('low','high')]
                self.cache[index]=(low,high)
                image=torch.from_numpy(np.ascontiguousarray(low)).permute(2,0,1).float()[None]/255
                image=image.contiguous(memory_format=torch.channels_last)
                raw=teacher(image)
                calibrated=calibrate(raw,calibration_features(image,raw),config['calibration'])
                colored=preserve_color(calibrated,image,**config['rgb_preservation'])
                output=refine_luminance(colored,**config['luminance_refinement'])
                self.reference[index]=output[0].to(torch.float16).contiguous()
                self.context[index]=(luma(raw).mean().item(),luma(colored).mean().item())
                if (index+1)%50==0 or index+1==len(rows):print(f'Full-image teacher cache {index+1}/{len(rows)}',flush=True)
        self.cache_seconds=time.perf_counter()-started

    def __getitem__(self,index):
        low,high,name=super().__getitem__(index)
        top,left,turns,flip=self.crop_parameters(index)
        reference=self.reference[index][...,top:top+self.patch,left:left+self.patch]
        reference=torch.rot90(reference,turns,(-2,-1))
        if flip:reference=reference.flip(-1)
        means=[torch.full((1,self.patch,self.patch),value) for value in self.context[index]]
        target=torch.cat((high,reference.float(),*means),dim=0)
        return low,target,name


def context_student_output(model,low,config,raw_mean,color_mean):
    raw=model(low)
    features=list(stable_calibration_features(low,raw))
    features[0]=luma(raw)-raw_mean
    calibrated=calibrate(raw,features,config['calibration'])
    colored=preserve_color(calibrated,low,**config['rgb_preservation'])
    parameters=config['luminance_refinement']
    old_y=luma(colored)
    new_y=(color_mean+parameters['contrast']*(old_y-color_mean)+parameters['lift']).clamp(0,1)
    chroma=colored-old_y;chroma=chroma-luma(chroma)
    margin=torch.minimum(new_y,1-new_y).clamp(0,1/255)
    positive=torch.where(chroma>1e-8,(1-margin-new_y)/chroma.clamp_min(1e-8),torch.ones_like(chroma))
    negative=torch.where(chroma< -1e-8,(new_y-margin)/(-chroma).clamp_min(1e-8),torch.ones_like(chroma))
    scale=torch.minimum(positive,negative).amin(1,keepdim=True).clamp(0,1)
    return (new_y+scale*chroma).clamp(0,1)


def redistribution_deficit(actual,truth,prior):
    truth,prior=truth.detach(),prior.detach()
    ax,ay=differences(actual);tx,ty=differences(truth);rx,ry=differences(prior)
    norm=(tx.square()+ty.square()+1e-8).sqrt()
    support=((norm-.004)/.025).clamp(0,1).detach()
    projected=(ax*tx+ay*ty)/norm/(2**.5)
    ref_projected=((rx*tx+ry*ty)/norm).clamp_min(0)/(2**.5)
    actual_mass=(projected*support).mean((-2,-1))
    prior_mass=(ref_projected*support).mean((-2,-1))
    # The deficit is detached: gradients only increase the correctly directed GT projection.
    missing=F.relu(stable_edges(prior).mean((-2,-1))*1.003-stable_edges(actual).mean((-2,-1)).detach())
    active=(support.mean((-2,-1))>.01).detach()
    return (F.relu(prior_mass+missing-actual_mass)*active).mean()


def make_context_detail_loss(config):
    def objective(model,low,packed,recipe):
        high,reference=packed[:,:3],packed[:,3:6].detach()
        raw_mean=packed[:,6:7].mean((-2,-1),keepdim=True).detach()
        color_mean=packed[:,7:8].mean((-2,-1),keepdim=True).detach()
        prediction=context_student_output(model,low,config,raw_mean,color_mean)
        actual,truth,prior=[luma(x) for x in (prediction,high,reference)]
        target=guided_detail_target(truth,prior,recipe['detail_gain'])
        edges=clean_structure_terms(actual,truth,prior,recipe['aligned_gain'])
        signed=actual.new_zeros(())
        for scale in (1,2):
            a,g=[F.avg_pool2d(x,scale) for x in (actual,target)]
            for axis in (-1,-2):signed=signed+F.l1_loss(a.diff(dim=axis),g.diff(dim=axis))
        mse=(prediction-high).square().mean((1,2,3))
        prior_mse=(reference-high).square().mean((1,2,3))
        mse_floor=F.relu(mse-prior_mse).mean()
        ssim_current,ssim_reference=ssim(prediction,high),ssim(reference,high)
        ssim_floor=F.relu(ssim_reference-ssim_current).mean()
        saturation=lambda x:(x.amax(1)-x.amin(1))/x.amax(1).clamp_min(1e-8)
        sat_floor=F.relu((saturation(prediction)-saturation(high)).abs().mean()-(saturation(reference)-saturation(high)).abs().mean())
        current_color,prior_color=color_terms(prediction,high),color_terms(reference,high)
        color_floor=F.relu(current_color['chroma']-prior_color['chroma'])+F.relu(current_color['chromaticity']-prior_color['chromaticity'])
        pm,rm=actual.mean((-2,-1)),prior.mean((-2,-1))
        brightness=(F.relu(rm*1.002-pm)+F.relu(pm-rm*1.02)).mean()
        spread=lambda x:(x.var((-2,-1),unbiased=False)+1e-8).sqrt()
        textured=(spread(truth)>.02).float().detach()
        contrast=((F.relu(spread(prior)-spread(actual))+F.relu(spread(actual)-1.03*spread(prior)))*textured).mean()
        entropy,ref_entropy=soft_entropy(actual,bins=64),soft_entropy(prior,bins=64)
        entropy_floor=(F.relu(ref_entropy-entropy)*textured.flatten(1).mean(1)).mean()
        loss=(2*F.l1_loss(actual,target)+2*(1-ssim_current).mean()+4*ssim_floor+5*mse.mean()+60*mse_floor+8*color_floor+4*sat_floor
              +8*signed/4+recipe['projection_weight']*(edges['aligned_deficit']+edges['wrong_direction']+edges['perpendicular'])
              +recipe['redistribution_weight']*redistribution_deficit(actual,truth,prior)
              +recipe['noise_weight']*(edges['flat_gradient']+edges['flat_texture'])
              +160*brightness+20*contrast+4*entropy_floor)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-context detail loss')
        return loss
    return objective
