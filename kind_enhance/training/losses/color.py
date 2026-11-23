"""GT-supervised colour refinement, independent of the deployed inference path."""
import torch
from torch.nn import functional as F
from kind_enhance.inference.calibration import luma,calibrate
from kind_enhance.training.losses.base import refinement_terms
from kind_enhance.training.losses.contrast_gradient import stable_calibration_features,soft_entropy

def chroma(x):
    return x-luma(x)

def chromaticity(x):
    # A finite denominator for black pixels; no angle/acos singularity.
    return x/(x.sum(1,keepdim=True)+.02)

def color_terms(prediction,target):
    target=target.detach()
    return {'chroma':F.l1_loss(chroma(prediction),chroma(target)),
            'chromaticity':F.l1_loss(chromaticity(prediction),chromaticity(target))}

def make_color_loss(teacher,output_config):
    from kind_enhance.inference.pipeline import enhance
    def objective(model,low,high,recipe):
        terms,raw=refinement_terms(model,low,high,recipe,return_prediction=True)
        final=calibrate(raw,stable_calibration_features(low,raw),output_config['calibration'])
        with torch.no_grad():
            reference=enhance(teacher,low,output_config)
        actual,target=luma(final),luma(reference)
        final_color,raw_color=color_terms(final,high),color_terms(raw,high)
        loss=sum(terms.values())
        loss=loss+recipe['color_weight']*(final_color['chroma']+.5*raw_color['chroma'])
        loss=loss+recipe['ratio_weight']*final_color['chromaticity']
        # A one-channel L cannot restore RGB ratios lost in decomposition.
        low_r,_=model.decomposition(low)
        loss=loss+recipe['decomposition_color_weight']*color_terms(low_r,high)['chromaticity']
        # Preserve luminance rather than copying the teacher's incorrect RGB.
        loss=loss+recipe['luma_teacher_weight']*F.l1_loss(actual,target)
        loss=loss+12*F.relu((actual.mean((-2,-1))-target.mean((-2,-1))).abs()-.005*target.mean((-2,-1))).mean()
        loss=loss+.25*(soft_entropy(actual)-soft_entropy(target)).abs().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite colour loss')
        return loss
    return objective
