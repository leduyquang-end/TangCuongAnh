import torch
from torch import nn
from torch.nn import functional as F


def double_conv(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.LeakyReLU(0.2),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.LeakyReLU(0.2),
    )


class AttentionDecoder(nn.Module):
    def __init__(self, inputs, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(inputs, channels, 2, stride=2)
        self.skip_projection = nn.Conv2d(channels, channels // 2, 1)
        self.gate_projection = nn.Conv2d(channels, channels // 2, 1)
        self.score = nn.Conv2d(channels // 2, 1, 1)
        self.fuse = double_conv(2 * channels, channels)

    def forward(self, features, skip):
        up = self.up(features)[..., :skip.shape[-2], :skip.shape[-1]]
        mask = self.score(F.relu(self.skip_projection(skip) + self.gate_projection(up))).sigmoid()
        return self.fuse(torch.cat((up, skip * mask), 1))


class AttentionUNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        channels = [width * 2**level for level in range(5)]
        self.encoders = nn.ModuleList(
            double_conv(inputs, outputs) for inputs, outputs in zip([4] + channels[:-1], channels)
        )
        self.decoders = nn.ModuleList(
            AttentionDecoder(inputs, outputs) for inputs, outputs in zip(channels[:0:-1], channels[-2::-1])
        )
        self.head = nn.Conv2d(width, 3, 1)

    def forward(self, reflectance, illumination):
        features = torch.cat((reflectance, illumination), 1)
        skips = []
        for encoder in self.encoders[:-1]:
            features = encoder(features)
            skips.append(features)
            features = F.max_pool2d(features, 2, ceil_mode=True)
        features = self.encoders[-1](features)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            features = decoder(features, skip)
        return self.head(features).sigmoid()
