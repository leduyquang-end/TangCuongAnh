from torch import nn

from .attention_unet import AttentionUNet
from .curve_cnn import CurveCNN
from .decomposition import KinDDecomposition


class KinDAttentionCurve(nn.Module):
    def __init__(self):
        super().__init__()
        self.decomposition = KinDDecomposition()
        self.restoration = AttentionUNet()
        self.adjustment = CurveCNN()

    def enhance(self, reflectance, illumination):
        restored = self.restoration(reflectance, illumination)
        adjusted, coefficients = self.adjustment(illumination)
        return restored * adjusted, restored, adjusted, coefficients

    def forward(self, image):
        return self.enhance(*self.decomposition(image))[0]
