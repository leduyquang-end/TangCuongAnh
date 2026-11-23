"""Differentiable contrast/edge targets with frozen-v5 brightness/entropy anchors."""
import torch
from torch.nn import functional as F
from kind_enhance.inference.calibration import luma
from kind_enhance.inference.calibration import calibration_features


def stable_calibration_features(low,prediction):
    """Identical forward calibration, finite derivative at zero edge norm."""
    _, detail, _ = calibration_features(low,prediction.detach())
    gray=luma(prediction)
    contrast=gray-gray.mean((-2,-1),keepdim=True)
    dx=F.pad(gray[...,1:]-gray[...,:-1],(0,1,0,0))
    dy=F.pad(gray[...,1:,:]-gray[...,:-1,:],(0,0,0,1))
    squared=dx.square()+dy.square()
    safe=squared.clamp_min(1e-12).sqrt()
    norm=safe+(squared.detach().sqrt()-safe.detach())
    importance=(norm/0.05).clamp(0,1)
    blur=F.avg_pool2d(prediction,3,1,1,count_include_pad=False)
    return contrast,detail,(prediction-blur)*importance


def soft_entropy(gray, bins=32):
    centers = torch.linspace(0, 1, bins, device=gray.device, dtype=gray.dtype)
    values = gray.flatten(1).unsqueeze(-1)
    memberships = (1 - (values - centers).abs() * (bins - 1)).clamp_min(0)
    probabilities = memberships.mean(1)
    probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-8)
    return -(probabilities * probabilities.clamp_min(1e-8).log2()).sum(1)


def ci_ag_terms(prediction, reference, clean):
    reference = reference.detach()
    gray, old, truth = [luma(image) for image in (prediction, reference, clean)]
    brightness = (gray.mean((-2,-1)) - old.mean((-2,-1))).abs().mean()
    entropy = (soft_entropy(gray) - soft_entropy(old)).abs().mean()
    def spread(image):
        return (image.var((-2,-1),unbiased=False)+1e-8).sqrt()-1e-4
    contrast = (spread(gray) - spread(old)*1.03).abs().mean()
    edges = gray.new_zeros(())
    for axis in (-1,-2):
        desired = truth.diff(dim=axis)
        previous = old.diff(dim=axis)
        support = (desired.abs()/0.04).clamp(0,1).detach()
        # Correct signed edges only; cap the requested amplitude by clean edges.
        amplitude = torch.maximum(desired.abs(), previous.abs()*1.25)
        amplitude = torch.minimum(amplitude, desired.abs()*1.25)
        goal = (desired.sign()*amplitude).detach()
        error = (gray.diff(dim=axis)-goal).abs()
        edges = edges + (error*support).sum()/support.sum().clamp_min(1)
    return {'brightness_anchor':brightness,'entropy_anchor':entropy,
            'ci_objective':contrast,'supported_edges':edges/2,
            'reference_pixels':F.l1_loss(prediction,reference)}


def continued_gain_terms(prediction, reference, clean):
    """Explicit CI/AG deficits; AG gain only on clean-supported gradients."""
    reference=reference.detach()
    gray,old,truth=[luma(image) for image in (prediction,reference,clean)]
    terms=ci_ag_terms(prediction,reference,clean)
    def spread(image):
        return (image.var((-2,-1),unbiased=False)+1e-8).sqrt()-1e-4
    current,prior=spread(gray),spread(old)
    terms['ci_objective']=(F.relu(1.05*prior-current)+F.relu(current-1.10*prior)).mean()
    def gradient_magnitude(image):
        origin=image[...,:-1,:-1]
        dx=image[...,:-1,1:]-origin
        dy=image[...,1:,:-1]-origin
        return ((dx.square()+dy.square())/2+1e-8).sqrt()-1e-4
    actual,previous,clean_edges=[gradient_magnitude(image) for image in (gray,old,truth)]
    support=(clean_edges/0.025).clamp(0,1).detach()
    supported=(actual*support).mean((-2,-1))
    prior_supported=(previous*support).mean((-2,-1))
    goal=(prior_supported+0.20*previous.mean((-2,-1))).detach()
    unsupported=(actual*(1-support)).mean((-2,-1))
    prior_unsupported=(previous*(1-support)).mean((-2,-1))
    # If no clean edges exist, do not ask for new texture or noise.
    has_support=(support.mean((-2,-1))>1e-5).detach()
    terms['supported_edges']=(F.relu(goal-supported)*has_support
                             +0.5*(unsupported-prior_unsupported).abs()).mean()
    return terms


def band_gain_terms(prediction, reference, clean):
    """Keep anchors within a band instead of pulling every patch to equality."""
    reference=reference.detach()
    terms=continued_gain_terms(prediction,reference,clean)
    gray,old=luma(prediction),luma(reference)
    mean,prior_mean=gray.mean((-2,-1)),old.mean((-2,-1))
    terms['brightness_anchor']=F.relu((mean-prior_mean).abs()-.005*prior_mean.abs()).mean()
    entropy,prior_entropy=soft_entropy(gray),soft_entropy(old)
    # Starting entropy is near the old-v5 upper tolerance. Allow a small
    # decrease in this proxy; do not reward additional entropy growth.
    terms['entropy_anchor']=(F.relu(entropy-prior_entropy)
                             +F.relu(prior_entropy*.995-entropy)).mean()
    return terms
