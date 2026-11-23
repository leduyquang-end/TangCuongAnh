"""Differentiable training proxies; validation always uses real uint8 metrics."""
import torch
from torch.nn import functional as F
from kind_enhance.training.losses.base import refinement_terms
from kind_enhance.training.losses.contrast_gradient import band_gain_terms, soft_entropy, stable_calibration_features
from kind_enhance.inference.calibration import calibrate, luma
from .policy import REFERENCE_STD


def edge_magnitude(gray):
    origin = gray[..., :-1, :-1]
    dx, dy = gray[..., :-1, 1:] - origin, gray[..., 1:, :-1] - origin
    return ((dx.square() + dy.square()) / 2 + 1e-8).sqrt() - 1e-4


def metric_proxies(prediction):
    gray = luma(prediction)
    mean = gray.mean((-3, -2, -1))
    spread = (gray.var((-3, -2, -1), unbiased=False) + 1e-8).sqrt() - 1e-4
    entropy = soft_entropy(gray, bins=32)
    edges = edge_magnitude(gray).mean((-3, -2, -1))
    return torch.stack((mean, spread, entropy, edges), dim=1)


def metric_reward(prediction, reference, component_weights):
    # Entropy32 is a differentiable proxy, not the reported entropy256 or official z.
    delta = metric_proxies(prediction) - metric_proxies(reference.detach()).detach()
    scales = prediction.new_tensor(REFERENCE_STD.tolist())
    weights = prediction.new_tensor(component_weights)
    return -((delta / scales) * weights).mean()


def make_loss(teacher, output_config):
    from kind_enhance.inference.pipeline import enhance
    def objective(model, low, high, recipe):
        with torch.no_grad():
            reference = enhance(teacher, low, output_config)
        terms, raw = refinement_terms(model, low, high, recipe, return_prediction=True)
        prediction = calibrate(raw, stable_calibration_features(low, raw), output_config['calibration'])
        extras = band_gain_terms(prediction, reference, high)
        loss = sum(terms.values()) + sum(recipe['extra_weights'][key] * value for key, value in extras.items())
        if recipe['metric_reward_weight']:
            loss = loss + recipe['metric_reward_weight'] * metric_reward(
                prediction, reference, recipe['metric_component_weights'])
        if recipe['flat_noise_weight']:
            actual = edge_magnitude(luma(prediction))
            previous = edge_magnitude(luma(reference))
            support = (edge_magnitude(luma(high)) / .025).clamp(0, 1).detach()
            loss = loss + recipe['flat_noise_weight'] * (F.relu(actual - previous) * (1 - support)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite engineering loss')
        return loss
    return objective
