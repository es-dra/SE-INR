"""
SC-INR Phase 2: Data-driven frequency basis omega_k(z).

Variant B (primary): omega(z) + phi=0
  - omega_conv(z) replaces fixed freqs buffer
  - Phase fixed at 0 (no phase_conv)
  - coef, sinc weights, MLP identical to Phase 1

Design:
  1. omega_conv: Conv2d(64, 2K, 3, pad=1) -> F.softplus -> [B, 2K, H, W]
  2. Per query: grid_sample omega_map -> [B, Q, K, 2] frequencies
  3. Fourier: cos/sin(pi * (omega_k * delta)) with phi=0
  4. Sinc: sinc(omega_k * c/2), analytic
  5. Amplitude modulation: coef * Fourier features
  6. MLP decoder -> RGB

FCE = 0 preserved: omega_conv input is z (not c), so F does not depend on c.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


def analytic_sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.sinc(x)


def init_log_polar_freqs(num_freqs, num_angles, freq_min, freq_max):
    """Same as Phase 1: generates reference frequencies for initialization."""
    assert num_freqs % num_angles == 0
    num_magnitudes = num_freqs // num_angles
    magnitudes = torch.logspace(
        math.log10(freq_min), math.log10(freq_max), num_magnitudes
    )
    angles = torch.linspace(0.0, math.pi, num_angles + 1)[:-1]
    freqs = []
    for mag in magnitudes:
        for theta in angles:
            freqs.append([mag.item() * math.cos(theta.item()),
                          mag.item() * math.sin(theta.item())])
    return torch.FloatTensor(freqs)  # [K, 2]


@register('sc_inr_adaptive')
@register('sc_inr_phase2')  # Backward compatibility for existing checkpoints.
class SCINRAdaptive(nn.Module):
    """
    SC-INR Phase 2: Data-driven omega_k(z) with phi=0.

    Variant B: omega(z) + phi=0 (clean theoretical design)
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
        local_ensemble: bool = True,
        upinput: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()

        assert hidden_dim % 2 == 0
        assert hidden_dim == 2 * num_freqs, (
            f"hidden_dim({hidden_dim}) must equal 2*num_freqs({2*num_freqs})"
        )

        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.local_ensemble = local_ensemble
        self.upinput = upinput

        self.encoder = models.make(encoder_spec)

        # Amplitude modulation (same as Phase 1)
        self.coef = nn.Conv2d(
            self.encoder.out_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )

        # Data-driven frequency basis: omega_k(z)
        # Output: [B, 2K, H, W] = 2 frequency components * K bases
        self.omega_conv = nn.Conv2d(
            self.encoder.out_dim, num_freqs * 2,
            kernel_size, padding=kernel_size // 2
        )

        # Initialize omega_conv: bias from Phase 1 freqs for warm start
        self._init_omega_conv(num_freqs, num_angles, freq_min, freq_max)

        # NOTE: No phase_conv. Phase is fixed at 0.
        # This forces all scale adaptation through the mathematically correct
        # channel: omega(z) -> sinc(omega * c/2).

        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def _init_omega_conv(self, num_freqs, num_angles, freq_min, freq_max):
        """Initialize omega_conv so epoch-0 behavior matches Phase 1."""
        ref_freqs = init_log_polar_freqs(num_freqs, num_angles, freq_min, freq_max)

        # Weight: small random, so initial output is dominated by bias
        nn.init.normal_(self.omega_conv.weight, mean=0.0, std=0.01)

        # Bias: set so that softplus(bias) ≈ ref_freqs at init
        # softplus(x) = log(1 + exp(x))
        # We want softplus(bias_k) = freqs_k -> bias_k = softplus_inverse(freqs_k)
        # For f >= 0: softplus_inverse(f) = log(exp(f) - 1)
        # For numerical safety when f is very small: use alternative
        with torch.no_grad():
            ref_flat = ref_freqs.flatten()  # [2K]
            # softplus_inverse via log(exp(f) - 1) with clamping
            bias_init = torch.where(
                ref_flat > 0.05,
                torch.log(torch.exp(ref_flat) - 1.0),  # accurate for f > 0.05
                ref_flat - 0.6931  # approximation for small f: softplus(x) ≈ x+ln2 near 0
            )
            self.omega_conv.bias.data = bias_init

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
        self.coeff = self.coef(self.feat)          # [B, hidden_dim, H, W]
        self.omega_map = F.softplus(self.omega_conv(self.feat))  # [B, 2K, H, W], positive

        return self.feat

    def query_rgb(self, coord: torch.Tensor, cell: torch.Tensor = None) -> torch.Tensor:
        feat = self.feat
        coef = self.coeff
        omega_map = self.omega_map

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

                q_coef = grid_fetch(coef)
                q_coord = grid_fetch(self.feat_coord)
                q_omega = grid_fetch(omega_map)    # [B, Q, 2K]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                bs, q = coord.shape[:2]
                K = self.num_freqs

                # Reshape omega to [B, Q, K, 2]
                q_omega = q_omega.view(bs, q, K, 2)

                # Step 1: omega_k · delta (phase=0, no phi term)
                q_phase = torch.sum(q_omega * rel_coord.unsqueeze(-2), dim=-1)  # [B, Q, K]

                # Step 2: Fourier features (cos + sin, phi=0)
                fourier_cos = torch.cos(math.pi * q_phase)  # [B, Q, K]
                fourier_sin = torch.sin(math.pi * q_phase)  # [B, Q, K]

                # Step 3: analytic sinc weights W_k(c)
                c_x = rel_cell[:, :, 0:1]  # [B, Q, 1]
                c_y = rel_cell[:, :, 1:2]  # [B, Q, 1]
                omega_x = q_omega[:, :, :, 0]  # [B, Q, K]
                omega_y = q_omega[:, :, :, 1]  # [B, Q, K]

                sinc_x = analytic_sinc(omega_x * c_x / 2)  # [B, Q, K]
                sinc_y = analytic_sinc(omega_y * c_y / 2)  # [B, Q, K]
                W = sinc_x * sinc_y

                # Step 4: sinc-weighted Fourier features
                fourier_cos_w = fourier_cos * W
                fourier_sin_w = fourier_sin * W
                fourier_feats = torch.cat([fourier_cos_w, fourier_sin_w], dim=-1)

                # Step 5: amplitude modulation
                inp_imnet = q_coef * fourier_feats

                # Step 6: MLP decoder
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

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
