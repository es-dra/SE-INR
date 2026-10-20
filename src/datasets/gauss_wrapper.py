"""Gaussian-blur degradation for arbitrary scale: blur HR with sigma = 0.6*s HR pixels, then bicubic resize WITHOUT anti-aliasing."""
import math, random
import torch, torch.nn.functional as F
from datasets import register
from datasets import wrappers as W

SIGMA_PER_SCALE = 0.6

def gauss_down(img, size):
    if isinstance(size, int): size = (size, size)
    c, h, w = img.shape
    s = h / size[0]; sig = SIGMA_PER_SCALE * s
    k = int(math.ceil(3 * sig)) * 2 + 1
    x = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-x ** 2 / (2 * sig ** 2)); g = g / g.sum()
    t = img[None]
    t = F.pad(t, (k // 2, k // 2, 0, 0), mode='reflect'); t = F.conv2d(t, g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    t = F.pad(t, (0, 0, k // 2, k // 2), mode='reflect'); t = F.conv2d(t, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return F.interpolate(t, size=size, mode='bicubic', align_corners=False, antialias=False).clamp(0, 1)[0]

@register('sr-implicit-downsampled-gauss')
class SRImplicitDownsampledGauss(W.SRImplicitDownsampled):
    def __getitem__(self, idx):
        orig = W.resize_fn
        W.resize_fn = gauss_down
        try:
            return super().__getitem__(idx)
        finally:
            W.resize_fn = orig
