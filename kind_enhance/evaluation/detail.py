"""Measure authentic edges and flat-region noise beside the unchanged competition metrics."""
import statistics
import torch
from torch.nn import functional as F
from kind_enhance.training.losses.supported_edges import stable_edges
from kind_enhance.inference.pipeline import enhance
from kind_enhance.evaluation.experiments import describe_color,means
from kind_enhance.inference.calibration import luma

def edge_diagnostics(prediction, target):
    actual, truth = luma(prediction), luma(target)
    ax, tx = [x[..., :-1, 1:] - x[..., :-1, :-1] for x in (actual, truth)]
    ay, ty = [x[..., 1:, :-1] - x[..., :-1, :-1] for x in (actual, truth)]
    norm_actual = (ax.square()+ay.square()).sqrt()
    norm_truth = (tx.square()+ty.square()).sqrt()
    support = (norm_truth > .01).float()
    cosine = ((ax*tx+ay*ty)/(norm_actual*norm_truth).clamp_min(1e-8)).clamp(-1,1)
    error = (ax-tx).abs()+(ay-ty).abs()
    flat = (stable_edges(truth)<.003).float()
    return dict(gradient_cosine=((cosine*support).sum()/support.sum().clamp_min(1)).item(),
                supported_gradient_mae=((error*support).sum()/(2*support.sum().clamp_min(1))).item(),
                GT_flat_fraction=flat.mean().item(),
                flat_AG_contribution=(stable_edges(actual)*flat).mean().item(),
                nonflat_AG_contribution=(stable_edges(actual)*(1-flat)).mean().item())


def extra_detail_diagnostics(prediction,target):
    actual,truth=luma(prediction),luma(target)
    origin=actual[...,:-1,:-1];gt_origin=truth[...,:-1,:-1]
    ax,ay=actual[...,:-1,1:]-origin,actual[...,1:,:-1]-origin
    tx,ty=truth[...,:-1,1:]-gt_origin,truth[...,1:,:-1]-gt_origin
    norm=(tx.square()+ty.square()+1e-8).sqrt()
    support=(norm>.01).float()
    projection=((ax*tx+ay*ty)/norm).clamp_min(0)
    flat=(stable_edges(truth)<.003).float()
    def highpass(x):return x-F.avg_pool2d(x,3,1,1,count_include_pad=False)
    residual=(highpass(actual)-highpass(truth))[...,:-1,:-1]
    return dict(aligned_AG=(projection*support).sum().div(support.sum().clamp_min(1)).item(),
                flat_residual_rms=((residual.square()*flat).sum()/flat.sum().clamp_min(1)).sqrt().item(),
                flat_ag=((stable_edges(actual)*flat).sum()/flat.sum().clamp_min(1)).item())


@torch.inference_mode()
def measure_detail_quality(model,dataset,config):
    model.eval();rows=[]
    for low,high,name in dataset:
        low,high=[x[None].contiguous(memory_format=torch.channels_last) for x in (low,high)]
        prediction=enhance(model,low,config)
        rows.append({**describe_color(prediction,high,name),**edge_diagnostics(prediction,high),**extra_detail_diagnostics(prediction,high)})
    result=means(rows)
    for key in rows[0]:
        if key!='id' and key not in result:result[key]=statistics.mean(row[key] for row in rows)
    return result,rows


def detail_quality_failures(metrics,baseline):
    # Numerical tolerances only; no percentage reduction allowance for preserved metrics.
    failures=[];violations={}
    floors={key:baseline[key] for key in ('MLI','CI','E','PSNR','SSIM','gradient_cosine','aligned_AG')}
    floors.update(AG=baseline['AG']*1.001,SSIM=baseline['SSIM']+.0001,
                  gradient_cosine=baseline['gradient_cosine']+.001,aligned_AG=baseline['aligned_AG']*1.01)
    ceilings={key:baseline[key] for key in ('chroma_mae','chromaticity_mae','saturation_mae','supported_gradient_mae','clipped_fraction')}
    ceilings.update(flat_ag=baseline['flat_ag']*.95,flat_residual_rms=baseline['flat_residual_rms']*.95)
    import math
    for key,floor in floors.items():
        delta=floor-metrics[key]
        if not math.isfinite(metrics[key]) or delta>1e-8:
            failures.append(key);violations[key]=max(delta,0)/max(abs(baseline[key]),.001) if math.isfinite(metrics[key]) else 1e6
    for key,ceiling in ceilings.items():
        delta=metrics[key]-ceiling
        if not math.isfinite(metrics[key]) or delta>1e-8:
            failures.append(key);violations[key]=max(delta,0)/max(abs(baseline[key]),.001) if math.isfinite(metrics[key]) else 1e6
    return failures,violations
