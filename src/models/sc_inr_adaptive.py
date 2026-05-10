"""
SC-INR Phase 2: Data-driven frequency basis omega_k(z).

Legacy no-phase variant: omega(z) + phi=0
  - omega_conv(z) replaces fixed freqs buffer
  - Phase fixed at 0 (no phase_conv)
  - coef, sinc weights, MLP identical to Phase 1

Design:
  1. omega_conv: Conv2d(64, 2K, 3, pad=1) -> omega parameterization -> [B, 2K, H, W]
  2. Per query: grid_sample omega_map -> [B, Q, K, 2] frequencies
  3. Fourier: cos/sin(pi * (omega_k * delta)) with phi=0
  4. Sinc: sinc(omega_k * c/2), analytic
  5. Amplitude modulation: coef * Fourier features
  6. MLP decoder -> RGB

Final-candidate variant: omega(z) + phi(z)
  - phase_conv(z) predicts K phase offsets independent of cell/scale
  - cell still enters only through analytic sinc response
  - the default remains phi=0 for checkpoint compatibility

NoSinc ablation: omega(z) + phi(z), without analytic response
  - keeps signed omega and feature-conditioned phase
  - removes only W(omega, c) so the Fourier observation is unweighted by cell

FCE = 0 preserved: omega_conv input is z (not c), so F does not depend on c.

Compatibility note:
  - sc_inr_adaptive is the legacy registry name for the paper display
    variant SC-INR-NoPhi.
  - sc_inr_adaptive_signed is the legacy registry name for SC-INR-NoPhi-Signed.
  - sc_inr_signed_phiz is the current final-candidate SC-INR.
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
    SC-INR-NoPhi: data-driven omega_k(z) with phi=0.

    The class name is kept for checkpoint compatibility.
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
        omega_param: str = "softplus",
        omega_bound: float | None = None,
        learn_phase: bool = False,
        phase_kernel_size: int = 1,
        phase_bias: bool = False,
        phase_zero_init: bool = True,
        use_sinc_response: bool = True,
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
        self.num_angles = num_angles
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.omega_param = omega_param
        self.omega_bound = omega_bound
        self.learn_phase = learn_phase
        self.phase_kernel_size = phase_kernel_size
        self.phase_bias = phase_bias
        self.phase_zero_init = phase_zero_init
        self.use_sinc_response = use_sinc_response
        self.local_ensemble = local_ensemble
        self.upinput = upinput

        if self.omega_param not in {"softplus", "tanh_signed"}:
            raise ValueError(f"Unsupported omega_param: {self.omega_param}")
        if self.omega_param == "tanh_signed":
            if self.omega_bound is None:
                self.omega_bound = float(freq_max) * 1.05
            if self.omega_bound <= 0:
                raise ValueError("omega_bound must be positive for tanh_signed omega")

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

        # Optional feature-conditioned phase. This is a content phase phi(z),
        # not LTE's cell-conditioned phase h_p(c).
        if learn_phase:
            self.phase_conv = nn.Conv2d(
                self.encoder.out_dim,
                num_freqs,
                phase_kernel_size,
                padding=phase_kernel_size // 2,
                bias=phase_bias,
            )
            if phase_zero_init:
                nn.init.zeros_(self.phase_conv.weight)
                if self.phase_conv.bias is not None:
                    nn.init.zeros_(self.phase_conv.bias)
        else:
            self.phase_conv = None

        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def _init_omega_conv(self, num_freqs, num_angles, freq_min, freq_max):
        """Initialize omega_conv so epoch-0 behavior matches Phase 1."""
        ref_freqs = init_log_polar_freqs(num_freqs, num_angles, freq_min, freq_max)

        with torch.no_grad():
            ref_flat = ref_freqs.flatten()  # [2K]
            if self.omega_param == "softplus":
                # Preserve the original Phase-2 behavior for existing checkpoints.
                nn.init.normal_(self.omega_conv.weight, mean=0.0, std=0.01)
                bias_init = torch.where(
                    ref_flat > 0.05,
                    torch.log(torch.exp(ref_flat) - 1.0),
                    ref_flat - 0.6931,
                )
            else:
                # Signed bounded omega: omega = omega_bound * tanh(raw).
                # Zero weights give an exact log-polar initialization at epoch 0.
                bound = float(self.omega_bound)
                if float(ref_flat.abs().max()) >= bound:
                    raise ValueError(
                        f"omega_bound ({bound}) must be larger than max |ref_freq| "
                        f"({float(ref_flat.abs().max())})"
                    )
                nn.init.zeros_(self.omega_conv.weight)
                normalized = (ref_flat / bound).clamp(-1 + 1e-6, 1 - 1e-6)
                bias_init = torch.atanh(normalized)
            self.omega_conv.bias.data = bias_init

    def _parameterize_omega(self, raw_omega: torch.Tensor) -> torch.Tensor:
        if self.omega_param == "softplus":
            return F.softplus(raw_omega)
        if self.omega_param == "tanh_signed":
            return float(self.omega_bound) * torch.tanh(raw_omega)
        raise RuntimeError(f"Unsupported omega_param: {self.omega_param}")

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
        self.omega_map = self._parameterize_omega(self.omega_conv(self.feat))  # [B, 2K, H, W]
        self.phase_map = self.phase_conv(self.feat) if self.phase_conv is not None else None

        return self.feat

    def query_rgb(self, coord: torch.Tensor, cell: torch.Tensor = None) -> torch.Tensor:
        feat = self.feat
        coef = self.coeff
        omega_map = self.omega_map
        phase_map = self.phase_map

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
                if phase_map is not None:
                    q_phase = q_phase + grid_fetch(phase_map)

                # Step 2: Fourier features (cos + sin)
                fourier_cos = torch.cos(math.pi * q_phase)  # [B, Q, K]
                fourier_sin = torch.sin(math.pi * q_phase)  # [B, Q, K]

                # Step 3: analytic sinc weights W_k(c), disabled only for
                # the SC-INR-NoSinc mechanism ablation.
                c_x = rel_cell[:, :, 0:1]  # [B, Q, 1]
                c_y = rel_cell[:, :, 1:2]  # [B, Q, 1]
                omega_x = q_omega[:, :, :, 0]  # [B, Q, K]
                omega_y = q_omega[:, :, :, 1]  # [B, Q, K]

                if self.use_sinc_response:
                    sinc_x = analytic_sinc(omega_x * c_x / 2)  # [B, Q, K]
                    sinc_y = analytic_sinc(omega_y * c_y / 2)  # [B, Q, K]
                    W = sinc_x * sinc_y
                else:
                    W = torch.ones_like(q_phase)

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


@register('sc_inr_adaptive_signed')
class SCINRAdaptiveSigned(SCINRAdaptive):
    """SC-INR-NoPhi-Signed: signed bounded omega variant.

    This class intentionally uses a separate registry name so that existing
    sc_inr_adaptive checkpoints keep their original softplus-positive omega
    semantics.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("omega_param", "tanh_signed")
        super().__init__(*args, **kwargs)


@register('sc_inr_adaptive_phiz')
@register('sc_inr_phase2_phiz')
@register('sc_inr_phiz')
class SCINRAdaptivePhiZ(SCINRAdaptive):
    """Feature-conditioned phase variant.

    The phase branch predicts phi(z) only. It does not receive cell/scale, so
    the decoder-side sampling response remains the analytic sinc term.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("learn_phase", True)
        super().__init__(*args, **kwargs)


@register('sc_inr_signed_phiz')
class SCINRSignedPhiZ(SCINRAdaptivePhiZ):
    """Final-candidate SC-INR: signed omega plus feature-conditioned phase."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("omega_param", "tanh_signed")
        super().__init__(*args, **kwargs)


@register('sc_inr_nosinc')
class SCINRNoSinc(SCINRSignedPhiZ):
    """SC-INR-NoSinc: signed omega plus phi(z), without sinc response."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_sinc_response", False)
        super().__init__(*args, **kwargs)
