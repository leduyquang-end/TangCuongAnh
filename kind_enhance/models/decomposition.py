import torch
from torch import nn
from torch.nn import functional as F


class KinDDecomposition(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(3, 32, 3, padding=1)
        self.second = nn.Conv2d(32, 64, 3, padding=1)
        self.bottom = nn.Conv2d(64, 128, 3, padding=1)
        self.up_second = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.decode_second = nn.Conv2d(128, 64, 3, padding=1)
        self.up_first = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.decode_first = nn.Conv2d(64, 32, 3, padding=1)
        self.reflectance = nn.Conv2d(32, 3, 1)
        self.light_features = nn.Conv2d(32, 32, 3, padding=1)
        self.illumination = nn.Conv2d(64, 1, 1)
        for layer in self.children():
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, image):
        first = F.leaky_relu(self.first(image), 0.2)
        second = F.leaky_relu(self.second(F.max_pool2d(first, 2, ceil_mode=True)), 0.2)
        bottom = F.leaky_relu(self.bottom(F.max_pool2d(second, 2, ceil_mode=True)), 0.2)
        up_second = self.up_second(bottom)[..., :second.shape[-2], :second.shape[-1]]
        decoded = F.leaky_relu(self.decode_second(torch.cat((up_second, second), 1)), 0.2)
        up_first = self.up_first(decoded)[..., :first.shape[-2], :first.shape[-1]]
        decoded = F.leaky_relu(self.decode_first(torch.cat((up_first, first), 1)), 0.2)
        light = F.leaky_relu(self.light_features(first), 0.2)
        return self.reflectance(decoded).sigmoid(), self.illumination(torch.cat((light, decoded), 1)).sigmoid()
