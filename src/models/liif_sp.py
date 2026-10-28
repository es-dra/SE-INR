"""LIIF with scale-paired modulation of every decoder layer.

The native LIIF decoder sees the cell only as two extra input coordinates.
Here the relative cell (2/s per axis) additionally modulates each hidden
layer through feature-wise affine maps (FiLM). Modulation heads start at
zero, so the model is initialised exactly as native LIIF.
"""
import torch
import torch.nn as nn

import models
from models import register
from models.liif import LIIF


class FiLMMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, cond_dim=2, embed_dim=64):
        super().__init__()
        self.linears = nn.ModuleList()
        lastv = in_dim
        for hidden in hidden_list:
            self.linears.append(nn.Linear(lastv, hidden))
            lastv = hidden
        self.out = nn.Linear(lastv, out_dim)
        self.embed = nn.Sequential(nn.Linear(cond_dim, embed_dim), nn.ReLU(),
                                   nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(embed_dim, 2 * h) for h in hidden_list])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, cond):
        e = self.embed(cond)
        for lin, head in zip(self.linears, self.heads):
            gamma, beta = head(e).chunk(2, dim=-1)
            x = torch.relu(lin(x) * (1 + gamma) + beta)
        return self.out(x)


@register('liif_sp')
class LIIFScalePaired(LIIF):

    def __init__(self, encoder_spec, imnet_spec, embed_dim=64, **kwargs):
        super().__init__(encoder_spec, imnet_spec=imnet_spec, cell_decode=True, **kwargs)
        in_dim = self.imnet.layers[0].in_features
        hidden = [m.out_features for m in self.imnet.layers if isinstance(m, nn.Linear)][:-1]
        out_dim = self.imnet.layers[-1].out_features
        self.imnet = _CondWrapper(FiLMMLP(in_dim, out_dim, hidden, embed_dim=embed_dim))


class _CondWrapper(nn.Module):
    """Takes LIIF's concatenated decoder input; the last two entries are the relative cell."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        cond = torch.log(x[:, -2:].clamp_min(1e-4))
        return self.net(x, cond)
