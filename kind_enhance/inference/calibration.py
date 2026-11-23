import torch
from torch.nn import functional as F

def luma(image):
    weights = image.new_tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
    return (image * weights).sum(1, keepdim=True)

def calibration_features(low, prediction):
    gray = luma(prediction)
    contrast = gray - gray.mean(dim=(-2, -1), keepdim=True)
    exposure_corrected = luma(low.pow(0.4))
    filtered = F.avg_pool2d(exposure_corrected, 3, 1, 1, count_include_pad=False)
    detail = filtered - F.avg_pool2d(filtered, 7, 1, 3, count_include_pad=False)
    detail = detail.sign() * (detail.abs() - 0.004).clamp(0, 0.08)
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(dx**2 + dy**2)
    edge_weight = (grad / 0.05).clamp(0, 1)
    blur = F.avg_pool2d(prediction, 3, 1, 1, count_include_pad=False)
    edge_detail = (prediction - blur) * edge_weight
    return contrast, detail, edge_detail

def calibrate(prediction, features, settings):
    contrast, detail, edge_detail = features
    sharpen = settings.get("sharpen", 0.0)
    return (prediction + (settings["contrast"] - 1) * contrast + settings["detail"] * detail + sharpen * edge_detail).clamp(0, 1)
