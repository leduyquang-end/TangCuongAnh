"""Four-metric curriculum with fixed teacher anchors and clean-supported AG."""
import torch
from torch.nn import functional as F
from .ag_refinement import stable_edges, student_output
from .aligned_edge_loss import aligned_edge_terms
from .luminance_refinement import enhance_all_gain
from .losses_ci_ag import soft_entropy
from .metrics import ssim
from .postprocessing import luma


def supported_absolute_ag(prediction, target, reference, desired_ag):
    target, reference = target.detach(), reference.detach()
    tgx, rgx = [x[..., :-1, 1:] - x[..., :-1, :-1] for x in (target, reference)]
    tgy, rgy = [x[..., 1:, :-1] - x[..., :-1, :-1] for x in (target, reference)]
    agreement = (tgx * rgx + tgy * rgy > 0).detach()
    support = ((stable_edges(target) / .025).clamp(0, 1) * agreement).detach()
    current, prior = stable_edges(prediction), stable_edges(reference)
    prior_total = prior.mean((-2, -1))
    supported_prior = (prior * support).mean((-2, -1))
    goal = supported_prior + (desired_ag - prior_total).clamp_min(0)
    active = (support.mean((-2, -1)) > .01).detach()
    return (F.relu(goal - (current * support).mean((-2, -1))) * active).mean()


def four_metric_loss(teacher, output_config):
    def objective(model, low, high, recipe):
        prediction = student_output(model, low, output_config)
        with torch.no_grad():
            reference = enhance_all_gain(teacher, low, output_config)
        actual, truth, prior = [luma(x) for x in (prediction, high, reference)]
        edges = aligned_edge_terms(actual, truth, prior, recipe['edge_gain'])
        ag = supported_absolute_ag(actual, truth, prior, recipe['ag_target'])
        mean, prior_mean = actual.mean((-2, -1)), prior.mean((-2, -1))
        # Positive mean margin compensates patch/global drift without using test.
        brightness = (F.relu(1.005 * prior_mean - mean) + F.relu(mean - 1.02 * prior_mean)).mean()
        spread = lambda x:(x.var((-2, -1), unbiased=False) + 1e-8).sqrt()
        contrast = (F.relu(1.06 * spread(prior) - spread(actual)) + F.relu(spread(actual) - 1.15 * spread(prior))).mean()
        entropy, prior_entropy = soft_entropy(actual, bins=64), soft_entropy(prior, bins=64)
        entropy_deficit = F.relu(prior_entropy + .003 - entropy).mean()
        loss = (F.l1_loss(prediction, high) + (1 - ssim(prediction, high)).mean()
                + 10 * edges['signed_edges'] + 80 * ag + 12 * edges['flat_noise']
                + 100 * brightness + 8 * contrast + 4 * entropy_deficit
                + .2 * F.l1_loss(actual, prior))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite four-metric curriculum loss')
        return loss
    return objective
