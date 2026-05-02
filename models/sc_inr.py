"""
SC-INR: Scale-Consistent INR with fixed log-spaced frequency basis.

Theory:
  Continuous function F is strictly separated from sampling operation.
  pixel(δ, c) = Σ_k A_k · cos(π · ω_k · δ + φ_k) · W_k(c)
  where W_k(c) = sinc(ω_k^x · c_x / 2) · sinc(ω_k^y · c_y / 2)

Differences from LTE / LTE-NoC:
  - LTE    : c enters via h_p(c) (learned phase), OOD c causes extrapolation failure
  - LTE-NoC: phase=0, c removed from all params, no sampling correction
  - SC-INR : c enters ONLY via analytic sinc weights, mathematically defined for any OOD c

Design:
  1. Fixed frequencies (log-polar init), not data-driven
  2. Amplitude A_k and phase φ_k both estimated from z (no c dependence)
  3. Sinc weights computed analytically (no learned params)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


def analytic_sinc(x: torch.Tensor) -> torch.Tensor:
    """sinc(x) = sin(π·x) / (π·x), using torch.sinc which gives sin(π·x)/(π·x)."""
    return torch.sinc(x)


def init_log_polar_freqs(
    num_freqs: int,
    num_angles: int,
    freq_min: float,
    freq_max: float,
) -> torch.Tensor:
    """
    Initialize log-spaced polar frequency basis.

    Args:
        num_freqs:  total frequencies K = num_magnitudes × num_angles
        num_angles: number of angle directions, uniform in [0, π)
        freq_min:   minimum frequency magnitude (in LR pixel coordinates)
        freq_max:   maximum frequency magnitude

    Returns:
        freqs: [K, 2] tensor, to be registered as buffer
    """
    assert num_freqs % num_angles == 0
    num_magnitudes = num_freqs // num_angles

    magnitudes = torch.logspace(
        math.log10(freq_min), math.log10(freq_max), num_magnitudes
    )
    angles = torch.linspace(0.0, math.pi, num_angles + 1)[:-1]

    freqs = []
    for mag in magnitudes:
        for theta in angles:
            freqs.append([
                mag.item() * math.cos(theta.item()),
                mag.item() * math.sin(theta.item()),
            ])

    return torch.FloatTensor(freqs)  # [K, 2]


@register('sc_inr')
class SCINR(nn.Module):
    """
    SC-INR with fixed log-spaced frequency basis and analytic sinc weights.

    Forward:
      z = Enc(LR)
      coef_conv(z) → q_coef  [bs, q, hidden_dim]
      phase_conv(z) → q_phi [bs, q, K]  (from z, no c)

      For each vx, vy (local ensemble):
        1. F_k(δ): cos(π·(ω_k·δ + φ_k)) + sin(π·(ω_k·δ + φ_k)) → 2K = hidden_dim
        2. W_k(c) = sinc(ω_k^x·c_x/2) · sinc(ω_k^y·c_y/2)
        3. inp = q_coef * [cos·W, sin·W]
        4. pred = MLP(inp) → [bs, q, 3]

      Local ensemble weighting + bilinear residual (upinput)
    """

    def __init__(
        self,
        encoder_spec,
        imnet_spec=None,
        hidden_dim: int = 256,
        num_freqs: int = 128,
        num_angles: int = 8,
        freq_min: float = 0.1,
        freq_max: float = 2.0,
        learn_phase: bool = True,
        local_ensemble: bool = True,
        upinput: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()

        assert hidden_dim % 2 == 0, "hidden_dim must be even (cos+sin halves)"
        assert hidden_dim == 2 * num_freqs, (
            f"hidden_dim({hidden_dim}) must equal 2*num_freqs({2*num_freqs})"
        )

        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        self.learn_phase = learn_phase
        self.local_ensemble = local_ensemble
        self.upinput = upinput

        self.encoder = models.make(encoder_spec)

        # Amplitude modulation conv (same role as LTE's coef conv)
        self.coef = nn.Conv2d(
            self.encoder.out_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )

        # Phase prediction conv (φ_k from z, no c dependence, replaces LTE's h_p(c))
        if learn_phase:
            self.phase_conv = nn.Conv2d(
                self.encoder.out_dim, num_freqs, kernel_size, padding=kernel_size // 2
            )
        else:
            self.phase_conv = None

        # Fixed frequency basis (log-polar, registered as buffer, no gradient)
        freqs = init_log_polar_freqs(num_freqs, num_angles, freq_min, freq_max)
        self.register_buffer('freqs', freqs)  # [K, 2], fixed

        # MLP decoder (same interface as LTE)
        assert imnet_spec is not None, "imnet_spec is required"
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def gen_feat(self, inp: torch.Tensor) -> torch.Tensor:
        self.inp = inp
        device = inp.device

        self.feat_coord = (
            make_coord(inp.shape[-2:], flatten=False)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(inp.shape[0], 2, *inp.shape[-2:])
        )

        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)  # [B, hidden_dim, H, W]

        if self.learn_phase:
            self.phase_map = self.phase_conv(self.feat)  # [B, K, H, W]
        else:
            self.phase_map = None

        return self.feat

    def query_rgb(self, coord: torch.Tensor, cell: torch.Tensor = None) -> torch.Tensor:
        feat = self.feat
        coef = self.coeff

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                def grid_fetch(feature_map):
                    return (
                        F.grid_sample(
                            feature_map,
                            coord_.flip(-1).unsqueeze(1),
                            mode='nearest',
                            align_corners=False,
                        )[:, :, 0, :]
                        .permute(0, 2, 1)
                    )

                q_coef = grid_fetch(coef)             # [bs, q, hidden_dim]
                q_coord = grid_fetch(self.feat_coord)  # [bs, q, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                bs, q = coord.shape[:2]
                K = self.num_freqs

                # Step 1: ω_k · δ  (no c dependence)
                q_phase = torch.matmul(rel_coord, self.freqs.T)  # [bs, q, K]

                # Step 2: add learned phase φ_k from z (no c)
                if self.learn_phase and self.phase_map is not None:
                    q_phi = grid_fetch(self.phase_map)  # [bs, q, K]
                    q_phase = q_phase + q_phi

                # Step 3: Fourier features (cos + sin)
                fourier_cos = torch.cos(math.pi * q_phase)  # [bs, q, K]
                fourier_sin = torch.sin(math.pi * q_phase)  # [bs, q, K]

                # Step 4: analytic sinc weights W_k(c)
                # c enters ONLY here, with no learned parameters
                c_x = rel_cell[:, :, 0:1]  # [bs, q, 1]
                c_y = rel_cell[:, :, 1:2]  # [bs, q, 1]
                fx = self.freqs[:, 0]       # [K]
                fy = self.freqs[:, 1]       # [K]

                sinc_x = analytic_sinc(fx * c_x / 2)  # [bs, q, K]
                sinc_y = analytic_sinc(fy * c_y / 2)  # [bs, q, K]
                W = sinc_x * sinc_y                     # [bs, q, K]

                # Step 5: sinc-weighted Fourier features
                fourier_cos_w = fourier_cos * W  # [bs, q, K]
                fourier_sin_w = fourier_sin * W  # [bs, q, K]

                # Concatenate to hidden_dim = 2K
                fourier_feats = torch.cat([fourier_cos_w, fourier_sin_w], dim=-1)
                # [bs, q, 2K] = [bs, q, hidden_dim]

                # Step 6: amplitude modulation (coef element-wise)
                inp_imnet = q_coef * fourier_feats  # [bs, q, hidden_dim]

                # Step 7: MLP decoder
                pred = self.imnet(
                    inp_imnet.contiguous().view(bs * q, -1)
                ).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        # Local ensemble weighted merge
        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        # Bilinear residual (upinput, same as LTE)
        if self.upinput:
            ret += (
                F.grid_sample(
                    self.inp,
                    coord.flip(-1).unsqueeze(1),
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False,
                )[:, :, 0, :]
                .permute(0, 2, 1)
            )

        return ret

    def forward(self, inp: torch.Tensor, coord: torch.Tensor, cell: torch.Tensor):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
