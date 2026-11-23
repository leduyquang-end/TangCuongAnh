"""Boost existing, GT-confirmed edge directions without adding a network."""
import torch
from torch.nn import functional as F
from .ag_refinement import stable_edges, student_output
from .luminance_refinement import enhance_all_gain
from .losses_ci_ag import soft_entropy
from .metrics import ssim
from .postprocessing import luma


def aligned_edge_terms(prediction, target, reference, gain):
    target, reference = target.detach(), reference.detach()
    signed_loss = prediction.new_zeros(())
    flat_loss = prediction.new_zeros(())
    for scale in (1, 2):
        actual, truth, prior = [F.avg_pool2d(x, scale) for x in (prediction, target, reference)]
        for axis in (-1, -2):
            pg, tg, rg = [x.diff(dim=axis) for x in (actual, truth, prior)]
            support = ((tg.abs() / .025).clamp(0, 1) * (tg * rg > 0)).detach()
            desired = (rg * (1 + (gain - 1) * support)).clamp(-.25, .25)
            weighted = ((pg - desired).abs() * support).sum((-2, -1))
            signed_loss = signed_loss + (weighted / support.sum((-2, -1)).clamp_min(1)).mean()
            flat_loss = flat_loss + (F.relu(pg.abs() - rg.abs()) * (1 - support)).mean()
    tgx, rgx = [x[..., :-1, 1:] - x[..., :-1, :-1] for x in (target, reference)]
    tgy, rgy = [x[..., 1:, :-1] - x[..., :-1, :-1] for x in (target, reference)]
    agreement = (tgx * rgx + tgy * rgy > 0).detach()
    support = ((stable_edges(target) / .025).clamp(0, 1) * agreement).detach()
    actual, prior = stable_edges(prediction), stable_edges(reference)
    has_edges = (support.mean((-2, -1)) > .01).detach()
    # Demand a 12% total AG increment only in GT-supported regions.
    goal = (prior * support).mean((-2, -1)) + .12 * prior.mean((-2, -1))
    deficit = (F.relu(goal - (actual * support).mean((-2, -1))) * has_edges).mean()
    return {'signed_edges': signed_loss / 4, 'flat_noise': flat_loss / 4, 'ag_deficit': deficit}


def make_aligned_loss(teacher, output_config):
    def objective(model, low, high, recipe):
        prediction = student_output(model, low, output_config)
        with torch.no_grad():
            reference = enhance_all_gain(teacher, low, output_config)
        actual, truth, prior = [luma(x) for x in (prediction, high, reference)]
        terms = aligned_edge_terms(actual, truth, prior, recipe['edge_gain'])
        mean, prior_mean = actual.mean((-2, -1)), prior.mean((-2, -1))
        spread = lambda x: (x.var((-2, -1), unbiased=False) + 1e-8).sqrt()
        ci = F.relu(recipe['ci_gain'] * spread(prior) - spread(actual)).mean()
        ci = ci + F.relu(spread(actual) - 1.12 * spread(prior)).mean()
        brightness = (F.relu(prior_mean - mean) + F.relu(mean - 1.01 * prior_mean)).mean()
        entropy, prior_entropy = soft_entropy(actual), soft_entropy(prior)
        entropy_loss = (F.relu(prior_entropy - entropy) + F.relu(entropy - 1.01 * prior_entropy)).mean()
        loss = (F.l1_loss(prediction, high) + .25 * (1 - ssim(prediction, high)).mean()
                + recipe['edge_weight'] * terms['signed_edges']
                + recipe['ag_weight'] * terms['ag_deficit']
                + recipe['noise_weight'] * terms['flat_noise']
                + recipe['ci_weight'] * ci + recipe['luma_anchor'] * F.l1_loss(actual, prior)
                + 12 * brightness + .5 * entropy_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite aligned edge loss')
        return loss
    return objective
