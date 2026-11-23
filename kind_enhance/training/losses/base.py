import torch
from torch.nn import functional as F

from kind_enhance.evaluation.reference import ssim


def kind_gradient(image, axis):
    """Valid differences, normalized independently for each image/channel."""
    if axis not in (-1, -2):
        raise ValueError("Expected a spatial axis")
    gradient = image.diff(dim=axis).abs()
    maximum = gradient.amax(dim=(-2, -1), keepdim=True)
    return gradient / (maximum + 0.0001)


def decomposition_loss(low, high, low_parts, high_parts):
    low_reflectance, low_light = low_parts
    high_reflectance, high_light = high_parts
    reconstruction = F.l1_loss(low_reflectance * low_light, low)
    reconstruction = reconstruction + F.l1_loss(high_reflectance * high_light, high)
    loss = reconstruction + 0.01 * F.l1_loss(low_reflectance, high_reflectance)
    weights = low.new_tensor([0.2989, 0.5870, 0.1140]).reshape(1, 3, 1, 1)
    low_gray, high_gray = (low * weights).sum(1, keepdim=True), (high * weights).sum(1, keepdim=True)
    for axis in (-1, -2):
        low_edge, high_edge = kind_gradient(low_light, axis), kind_gradient(high_light, axis)
        combined = low_edge + high_edge
        loss = loss + 0.2 * (combined * (-10 * combined).exp()).mean()
        loss = loss + 0.15 * (low_edge / kind_gradient(low_gray, axis).clamp_min(0.01)).mean()
        loss = loss + 0.15 * (high_edge / kind_gradient(high_gray, axis).clamp_min(0.01)).mean()
    return loss


def compute_loss(model, low, high, stage, structure):
    if stage == "decomp":
        return decomposition_loss(low, high, model.decomposition(low), model.decomposition(high))
    with torch.no_grad():
        low_parts, (target_reflectance, target_light) = model.decomposition(low), model.decomposition(high)
    prediction, restored, adjusted, coefficients = model.enhance(*low_parts)
    smooth = coefficients.diff(dim=-1).abs().mean() + coefficients.diff(dim=-2).abs().mean()
    return (F.l1_loss(prediction, high) + structure * (1 - ssim(prediction, high)).mean()
            + 0.10 * F.l1_loss(restored, target_reflectance)
            + 0.10 * F.l1_loss(adjusted, target_light) + 0.01 * smooth)


def refinement_loss(model, low, high, config):
    return sum(refinement_terms(model, low, high, config).values())


def refinement_terms(model, low, high, config, return_prediction=False):
    """Weighted terms for training and diagnostics; targets remain unamplified."""
    low_parts = model.decomposition(low)
    high_parts = model.decomposition(high)
    prediction, reflectance, illumination, coefficients = model.enhance(*low_parts)
    gradient_loss = sum(F.l1_loss(prediction.diff(dim=axis), high.diff(dim=axis)) for axis in (-1, -2))
    curve_smoothness = sum(coefficients.diff(dim=axis).abs().mean() for axis in (-1, -2))
    terms = {
        "rgb_l1": F.l1_loss(prediction, high),
        "structure": config["structure"] * (1 - ssim(prediction, high)).mean(),
        "rgb_gradient": config["gradient_weight"] * gradient_loss,
        "reflectance": 0.10 * F.l1_loss(reflectance, high_parts[0].detach()),
        "illumination": 0.10 * F.l1_loss(illumination, high_parts[1].detach()),
        "curve_smoothness": 0.01 * curve_smoothness,
        "decomposition": config["decomposition_weight"] * decomposition_loss(low, high, low_parts, high_parts),
    }
    if config.get("detail_weight") or config.get("contrast_weight"):
        terms["contrast_detail"] = contrast_detail_loss(prediction, high, config)
    if config.get("edge_alignment_weight"):
        terms["edge_alignment"] = edge_alignment_loss(prediction, high, config["edge_alignment_weight"])
    return (terms, prediction) if return_prediction else terms


def contrast_detail_loss(prediction, target, config):
    weights = prediction.new_tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
    predicted_gray, target_gray = (tensor.mul(weights).sum(1, keepdim=True) for tensor in (prediction, target))
    predicted_detail = predicted_gray - F.avg_pool2d(predicted_gray, 5, 1, 2, count_include_pad=False)
    target_detail = target_gray - F.avg_pool2d(target_gray, 5, 1, 2, count_include_pad=False)
    detail = F.l1_loss(predicted_detail, target_detail)

    def contrast(image, window):
        # Centered variance avoids cancellation in E[x^2] - E[x]^2.
        window = min(window, *image.shape[-2:])
        patches = image.unfold(2, window, window).unfold(3, window, window)
        return (patches.var(dim=(-2, -1), unbiased=False) + 1e-6).sqrt()

    local = F.l1_loss(contrast(predicted_gray, 16), contrast(target_gray, 16))
    predicted_std = (predicted_gray.var(dim=(-2, -1), unbiased=False) + 1e-6).sqrt()
    target_std = (target_gray.var(dim=(-2, -1), unbiased=False) + 1e-6).sqrt()
    global_contrast = F.l1_loss(predicted_std, target_std)
    return config.get("detail_weight", 0.0) * detail + config.get("contrast_weight", 0.0) * (local + global_contrast)


def edge_alignment_loss(prediction, target, weight):
    weights = prediction.new_tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
    predicted_gray, target_gray = (tensor.mul(weights).sum(1, keepdim=True) for tensor in (prediction, target))
    loss = predicted_gray.new_zeros(())
    for scale in (1, 2):
        predicted = F.avg_pool2d(predicted_gray, scale)
        reference = F.avg_pool2d(target_gray, scale)
        for axis in (-1, -2):
            target_edge = reference.diff(dim=axis)
            edge_strength = (target_edge.abs() / 0.05).clamp(0, 1)
            importance = (1 + 2 * edge_strength).detach()
            error = (predicted.diff(dim=axis) - target_edge).abs()
            # Normalize within each image: textured samples cannot dominate
            # another sample simply by containing more target edges.
            weighted_error = (error * importance).sum(dim=(-2, -1))
            loss = loss + (weighted_error / importance.sum(dim=(-2, -1))).mean()
    return weight * loss / 4


