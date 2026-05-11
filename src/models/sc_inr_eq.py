"""SC-INR-EQ: SC-INR decoder contract with Rot-E plumbing.

This exploratory model keeps the SC-INR sampling contract:
  - coef(z), omega(z), and phi(z) are feature-conditioned only;
  - cell/scale enters only through the analytic sinc response;
  - LTE-EQ's learned phase(cell) is intentionally not used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models import B_Conv as fn
from models.sc_inr_adaptive import analytic_sinc, init_log_polar_freqs
from utils import make_coord


@register("sc_inr_eq")
class SCINREQ(nn.Module):
    """Rot-E extension of final-candidate SC-INR.

    The equivariant channel layout follows LTE-EQ:
    omega is arranged as [2, K_per_transform, tranNum], while coef and phi
    use feature-major, transform-minor ordering.
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
        omega_bound: float = 2.1,
        learn_phase: bool = True,
        phase_kernel_size: int = 1,
        phase_bias: bool = False,
        phase_zero_init: bool = True,
        use_sinc_response: bool = True,
        local_ensemble: bool = True,
        upinput: bool = True,
        tranNum: int = 4,
        kernel_size: int = 5,
        corrd_scale: float = 1.0,
    ):
        super().__init__()

        if hidden_dim != 2 * num_freqs:
            raise ValueError(
                f"hidden_dim({hidden_dim}) must equal 2*num_freqs({2 * num_freqs})"
            )
        if num_freqs % tranNum != 0:
            raise ValueError("num_freqs must be divisible by tranNum")
        if hidden_dim % tranNum != 0:
            raise ValueError("hidden_dim must be divisible by tranNum")
        if omega_bound <= 0:
            raise ValueError("omega_bound must be positive")

        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        self.num_angles = num_angles
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.omega_bound = omega_bound
        self.learn_phase = learn_phase
        self.phase_kernel_size = phase_kernel_size
        self.phase_bias = phase_bias
        self.phase_zero_init = phase_zero_init
        self.use_sinc_response = use_sinc_response
        self.local_ensemble = local_ensemble
        self.upinput = upinput
        self.tranNum = tranNum
        self.kernel_size = kernel_size
        self.corrd_scale = corrd_scale
        self.num_freqs_per_tran = num_freqs // tranNum

        self.encoder = models.make(encoder_spec, args={"tranNum": tranNum})
        if self.encoder.out_dim % tranNum != 0:
            raise ValueError("equivariant encoder out_dim must be divisible by tranNum")

        in_dim_per_tran = self.encoder.out_dim // tranNum
        hidden_per_tran = hidden_dim // tranNum
        omega_per_tran = (2 * num_freqs) // tranNum
        phase_per_tran = num_freqs // tranNum

        self.coef = fn.Fconv_PCA(
            kernel_size,
            in_dim_per_tran,
            hidden_per_tran,
            tranNum=tranNum,
            padding=kernel_size // 2,
        )
        self.omega_conv = fn.Fconv_PCA(
            kernel_size,
            in_dim_per_tran,
            omega_per_tran,
            tranNum=tranNum,
            padding=kernel_size // 2,
        )
        self._init_omega_conv()

        if learn_phase:
            if phase_kernel_size == 1:
                self.phase_conv = fn.Fconv_1X1(
                    in_dim_per_tran,
                    phase_per_tran,
                    tranNum=tranNum,
                    bias=phase_bias,
                )
                if not phase_bias:
                    # Fconv_1X1 stores the no-bias zero tensor as a plain
                    # attribute; register it so .to(device) moves it.
                    delattr(self.phase_conv, "c")
                    self.phase_conv.register_buffer(
                        "c", torch.zeros(1, phase_per_tran, 1, 1)
                    )
            else:
                self.phase_conv = fn.Fconv_PCA(
                    phase_kernel_size,
                    in_dim_per_tran,
                    phase_per_tran,
                    tranNum=tranNum,
                    padding=phase_kernel_size // 2,
                    bias=phase_bias,
                )
            if phase_zero_init:
                nn.init.zeros_(self.phase_conv.weights)
                if getattr(self.phase_conv, "c", None) is not None:
                    nn.init.zeros_(self.phase_conv.c)
        else:
            self.phase_conv = None

        self.imnet = models.make(
            imnet_spec, args={"tranNum": tranNum, "in_dim": hidden_dim}
        )
        self._init_rotation_buffers()

    def _init_rotation_buffers(self) -> None:
        theta = torch.arange(self.tranNum) / self.tranNum * 2 * math.pi
        theta = -theta.reshape(1, 1, 1, self.tranNum)
        self.register_buffer("cosTheta", torch.cos(theta))
        self.register_buffer("sinTheta", torch.sin(theta))

    def _init_omega_conv(self) -> None:
        ref = init_log_polar_freqs(
            self.num_freqs_per_tran,
            self.num_angles,
            self.freq_min,
            self.freq_max,
        )
        ref_flat = ref.t().contiguous().view(-1)
        if float(ref_flat.abs().max()) >= float(self.omega_bound):
            raise ValueError(
                f"omega_bound ({self.omega_bound}) must be larger than max |ref_freq| "
                f"({float(ref_flat.abs().max())})"
            )

        with torch.no_grad():
            nn.init.zeros_(self.omega_conv.weights)
            normalized = (ref_flat / float(self.omega_bound)).clamp(
                -1 + 1e-6, 1 - 1e-6
            )
            self.omega_conv.c.copy_(torch.atanh(normalized).view(1, -1, 1, 1))

    def _parameterize_omega(self, raw_omega: torch.Tensor) -> torch.Tensor:
        return float(self.omega_bound) * torch.tanh(raw_omega)

    def gen_feat(self, inp: torch.Tensor) -> torch.Tensor:
        self.inp = inp
        self.feat_coord = (
            make_coord(inp.shape[-2:], flatten=False)
            .to(inp.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(inp.shape[0], 2, *inp.shape[-2:])
        )

        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.omega_map = self._parameterize_omega(self.omega_conv(self.feat))
        self.phase_map = self.phase_conv(self.feat) if self.phase_conv is not None else None
        return self.feat

    def _grid_fetch(self, feature_map: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        return (
            F.grid_sample(
                feature_map,
                coord.flip(-1).unsqueeze(1),
                mode="nearest",
                align_corners=False,
            )[:, :, 0, :]
            .permute(0, 2, 1)
        )

    def _basis_and_response(
        self,
        q_omega: torch.Tensor,
        q_phi: torch.Tensor | None,
        rel_coord: torch.Tensor,
        rel_cell: torch.Tensor,
    ) -> torch.Tensor:
        bs, q = rel_coord.shape[:2]
        k = self.num_freqs_per_tran
        t = self.tranNum

        omega = q_omega.view(bs, q, 2, k, t)

        x = rel_coord[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        y = rel_coord[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        rot_x = self.cosTheta * x - self.sinTheta * y
        rot_y = self.sinTheta * x + self.cosTheta * y
        rot_coord = torch.stack([rot_x, rot_y], dim=2)

        phase = torch.sum(omega * (rot_coord * self.corrd_scale), dim=2)
        phase = phase.reshape(bs, q, self.num_freqs)
        if q_phi is not None:
            phase = phase + q_phi

        fourier_cos = torch.cos(math.pi * phase)
        fourier_sin = torch.sin(math.pi * phase)

        if self.use_sinc_response:
            omega_x = omega[:, :, 0]
            omega_y = omega[:, :, 1]
            eff_x = self.cosTheta * omega_x + self.sinTheta * omega_y
            eff_y = -self.sinTheta * omega_x + self.cosTheta * omega_y
            if self.corrd_scale != 1.0:
                eff_x = eff_x * self.corrd_scale
                eff_y = eff_y * self.corrd_scale
            c_x = rel_cell[:, :, 0].unsqueeze(-1).unsqueeze(-1)
            c_y = rel_cell[:, :, 1].unsqueeze(-1).unsqueeze(-1)
            response = analytic_sinc(eff_x * c_x / 2) * analytic_sinc(eff_y * c_y / 2)
            response = response.reshape(bs, q, self.num_freqs)
        else:
            response = torch.ones_like(phase)

        return torch.cat([fourier_cos * response, fourier_sin * response], dim=-1)

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

                q_coef = self._grid_fetch(coef, coord_)
                q_omega = self._grid_fetch(omega_map, coord_)
                q_coord = self._grid_fetch(self.feat_coord, coord_)
                q_phi = self._grid_fetch(phase_map, coord_) if phase_map is not None else None

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                fourier_feats = self._basis_and_response(
                    q_omega, q_phi, rel_coord, rel_cell
                )
                inp_imnet = q_coef * fourier_feats

                bs, q = coord.shape[:2]
                pred = self.imnet(inp_imnet.contiguous().view(bs * q, -1)).view(
                    bs, q, -1
                )
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        if self.upinput:
            ret += (
                F.grid_sample(
                    self.inp,
                    coord.flip(-1).unsqueeze(1),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )[:, :, 0, :]
                .permute(0, 2, 1)
            )

        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
