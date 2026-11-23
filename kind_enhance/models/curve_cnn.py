import torch
from torch import nn
from torch.nn import functional as F


class CurveCNN(nn.Module):
    def __init__(self, width=16, steps=8):
        super().__init__()
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.encoder = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(3))
        self.decoder = nn.ModuleList(nn.Conv2d(2 * width, width, 3, padding=1) for _ in range(2))
        self.coefficients = nn.Conv2d(2 * width, steps, 3, padding=1)
        nn.init.normal_(self.coefficients.weight, std=0.001)
        nn.init.zeros_(self.coefficients.bias)

    def forward(self, illumination):
        features = F.relu(self.stem(illumination))
        skips = []
        for layer in self.encoder:
            skips.append(features)
            features = F.relu(layer(features))
        for layer, skip in zip(self.decoder, reversed(skips[1:])):
            features = F.relu(layer(torch.cat((features, skip), 1)))
        coefficients = self.coefficients(torch.cat((features, skips[0]), 1)).tanh()
        adjusted = illumination
        for coefficient in coefficients.split(1, dim=1):
            adjusted = adjusted + coefficient * adjusted * (1 - adjusted)
        return adjusted, coefficients
