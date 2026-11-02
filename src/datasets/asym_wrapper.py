"""Training wrapper with independent vertical and horizontal scales.

Each sample draws s_y, s_x ~ U(scale_min, scale_max) independently, crops an
HR window of round(inp_size*s_y) x round(inp_size*s_x) and resizes it to the
square inp_size x inp_size LR input, so the decoder sees cells (2/s_y, 2/s_x)
with c_y != c_x. Everything else (augmentation, query sampling, keys) follows
`sr-implicit-downsampled`; the `scale` entry is [s_y, s_x] after augmentation.
"""
import random

import numpy as np
import torch

from datasets import register
from datasets import wrappers as W
from utils import to_pixel_samples


@register('sr-implicit-downsampled-asym')
class SRImplicitDownsampledAsym(W.SRImplicitDownsampled):

    def __getitem__(self, idx):
        img = self.dataset[idx]
        sy = random.uniform(self.scale_min, self.scale_max)
        sx = random.uniform(self.scale_min, self.scale_max)
        assert self.inp_size is not None, 'asymmetric wrapper is for training crops'
        w_lr = self.inp_size
        h_hr, w_hr = round(w_lr * sy), round(w_lr * sx)
        x0 = random.randint(0, img.shape[-2] - h_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0:x0 + h_hr, y0:y0 + w_hr]
        crop_lr = W.resize_fn(crop_hr, (w_lr, w_lr))

        if self.augment:
            hflip, vflip, dflip = (random.random() < 0.5 for _ in range(3))

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr, crop_hr = augment(crop_lr), augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord, hr_rgb = hr_coord[sample_lst], hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        scale = torch.Tensor([crop_hr.shape[-2] / crop_lr.shape[-2], crop_hr.shape[-1] / crop_lr.shape[-1]])
        return {'inp': crop_lr, 'coord': hr_coord, 'cell': cell, 'scale': scale, 'gt': hr_rgb}
