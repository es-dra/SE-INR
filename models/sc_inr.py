"""
SC-INR: Sinc-weighted Convolutional Implicit Neural Representation

Core idea: Separate the continuous signal F(δ) from the sampling operation W_k(c).
- F(δ) = Σ A_k · cos(ω_k·δ + φ_k)  (no c dependence)
- pixel = F(δ) · sinc(ω_k·c/2)       (c only affects the weight, not F)

Phase 1 implementation with K=16 learnable frequency bases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import numpy as np


@register('sc_inr')
class SCINR(nn.Module):
    """
    SC-INR with K=16 learnable sine bases and sinc weighting.

    Forward formula:
        RGB(δ, c) = Σ_{k=1}^{K} A_k · cos(ω_k^x·δ_x + ω_k^y·δ_y + φ_k) · sinc(ω_k^x·c_x/2) · sinc(ω_k^y·c_y/2)

    MLP outputs: {A_k(3K), ω_k(2K), φ_k(K)} = 6K values
    """

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256,
                 local_ensemble=True, upinput=True, kernel_size=3, K=16):
        super().__init__()
        self.K = K
        self.local_ensemble = local_ensemble
        self.upinput = upinput

        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, kernel_size, padding=1)

        # MLP output: 3K(A) + 2K(ω) + K(φ) = 6K
        mlp_in_dim = hidden_dim + 2 + 2  # feat + rel_coord + rel_cell
        mlp_out_dim = 6 * K

        # Initialize log-omega bias for good frequency initialization
        log_omega_init = np.log(np.linspace(0.1, 4.0, K))  # (K,)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for A, omega, phi
        self.head_A = nn.Linear(hidden_dim, 3 * K)
        self.head_omega = nn.Linear(hidden_dim, 2 * K)
        self.head_phi = nn.Linear(hidden_dim, K)

        # Initialize omega head bias with log-omega values
        with torch.no_grad():
            self.head_omega.bias[::2].copy_(torch.tensor(log_omega_init))  # omega_x
            self.head_omega.bias[1::2].copy_(torch.tensor(log_omega_init))  # omega_y

    @staticmethod
    def sinc(x):
        """Sinc function: sin(πx)/(πx). Handles x=0 safely."""
        x = torch.clamp(x, min=1e-6, max=1e6)
        return torch.sin(np.pi * x) / (np.pi * x)

    def query_rgb(self, coord, cell=None):
        """Query RGB values at given coordinates."""
        feat = self.feat
        coef = self.coeff

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        eps_shift = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                bs, q = coord.shape[:2]
                inp = torch.cat([q_coef, rel_coord, rel_cell], dim=-1)

                # MLP forward
                h = self.mlp(inp.view(bs * q, -1))

                # Heads
                A = self.head_A(h)  # (B*Q, 3K)
                omega_log = self.head_omega(h)  # (B*Q, 2K)
                phi = self.head_phi(h)  # (B*Q, K)

                # Convert log-omega to omega (positive)
                omega = torch.exp(torch.clamp(omega_log, min=-10, max=10))

                # Split into x and y components
                omega_x = omega[:, :self.K]  # (B*Q, K)
                omega_y = omega[:, self.K:2 * self.K]  # (B*Q, K)

                # Reshape for broadcasting
                A = A.view(bs, q, self.K, 3)  # (B, Q, K, 3)
                omega_x = omega_x.view(bs, q, self.K)
                omega_y = omega_y.view(bs, q, self.K)
                phi = phi.view(bs, q, self.K)

                # Compute cos(ω·δ + φ)
                dot_product = (omega_x * rel_coord[:, :, 0:1] +
                               omega_y * rel_coord[:, :, 1:2])
                dot_product = dot_product + phi
                cos_basis = torch.cos(dot_product)

                # Compute sinc weights: sinc(ω·c/2)
                sinc_x = self.sinc(omega_x * rel_cell[:, :, 0:1] / 2)
                sinc_y = self.sinc(omega_y * rel_cell[:, :, 1:2] / 2)
                sinc_weight = sinc_x * sinc_y

                # RGB = Σ A_k · cos · sinc
                rgb = (A * cos_basis.unsqueeze(-1) * sinc_weight.unsqueeze(-1)).sum(dim=2)

                preds.append(rgb)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        if self.upinput:
            ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)

        return ret

    def gen_feat(self, inp):
        self.inp = inp
        device = inp.device
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).to(device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])

        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        return self.feat

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)