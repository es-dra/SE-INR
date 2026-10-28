"""Training wrapper that samples one contiguous HR window instead of random query points.

Returns the same number of queries as `sample_q = window**2`, laid out row-major,
so a frequency-domain loss can be computed on the window.
"""
import random

import torch

from datasets import register
from datasets import wrappers as W


@register('sr-implicit-downsampled-window')
class SRImplicitDownsampledWindow(W.SRImplicitDownsampled):

    def __init__(self, dataset, window=48, **kwargs):
        kwargs['sample_q'] = None
        super().__init__(dataset, **kwargs)
        self.window = window

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        side = round(item['coord'].shape[0] ** 0.5)
        w = self.window
        i0 = random.randint(0, side - w)
        j0 = random.randint(0, side - w)
        rows = torch.arange(i0, i0 + w)[:, None] * side
        index = (rows + torch.arange(j0, j0 + w)[None, :]).reshape(-1)
        for k in ('coord', 'cell', 'gt'):
            item[k] = item[k][index]
        return item
