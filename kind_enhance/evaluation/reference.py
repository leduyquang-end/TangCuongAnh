import math

import torch
from torch.nn import functional as F


def ssim(prediction, target):
    coordinate = torch.arange(-5, 6, device=prediction.device, dtype=prediction.dtype)
    gaussian = (-coordinate.square() / 4.5).exp()
    gaussian = gaussian / gaussian.sum()
    channels = prediction.shape[1]
    horizontal = gaussian.reshape(1, 1, 1, 11).expand(channels, 1, 1, 11).contiguous()
    vertical = horizontal.transpose(-1, -2).contiguous()
    moments = torch.cat((prediction, target, prediction.square(), target.square(), prediction * target))
    moments = F.conv2d(F.conv2d(moments, horizontal, groups=channels), vertical, groups=channels)
    mean_pred, mean_target, second_pred, second_target, cross = moments.chunk(5)
    covariance = cross - mean_pred * mean_target
    variance_pred = (second_pred - mean_pred.square()).clamp_min(0)
    variance_target = (second_target - mean_target.square()).clamp_min(0)
    luminance = (2 * mean_pred * mean_target + 0.0001) / (mean_pred.square() + mean_target.square() + 0.0001)
    contrast = (2 * covariance + 0.0009) / (variance_pred + variance_target + 0.0009)
    return (luminance * contrast).mean(dim=(1, 2, 3))


def score(prediction, target):
    return psnr_db(prediction, target), ssim(prediction, target).item()


def psnr_db(prediction, target):
    mse = F.mse_loss(prediction, target).item()
    return -10 * math.log10(max(mse, 1e-12))
