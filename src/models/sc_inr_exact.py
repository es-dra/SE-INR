"""SC-INR-exact: a local Fourier field whose output is the exact average over the query footprint.

Each LR feature cell (i, j) carries a local expansion
    f_ij(y, x) = sum_k a_k cos(2*pi*(w_ky (y - y_i) + w_kx (x - x_j)) + phi_k)
in LR-pixel units, and the continuous field is f(y, x) = f_{cell(y, x)}(y, x).
A query with centre (y_q, x_q) and footprint (c_y, c_x) LR pixels returns the
exact mean of f over its footprint rectangle. The rectangle is partitioned by
the feature grid; on each piece the integral of a sinusoid is closed form:
    mean over [ya, yb] x [xa, xb] of cos(A y + B x + phi)
      = sinc(A L_y / 2) sinc(B L_x / 2) cos(A y_m + B x_m + phi),
with L the piece side lengths and (y_m, x_m) its midpoint. The piece weights
L_y L_x / (c_y c_x) sum to one. Because the readout is linear in these
features, the output is exactly the footprint average, so a 2x2 box average of
the s-scale output equals the s/2-scale output, and asymmetric footprints,
downscaling (c > 1) and any out-of-range cell are handled by the same formula.

A low-frequency base term is the exact footprint average of the bilinear
interpolant of the LR image (piecewise bilinear on the dual grid, whose
rectangle mean equals its midpoint value). `base='none'` removes it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models.sc_inr_adaptive import init_log_polar_freqs


def _partition(t_q, half, n, origin):
    """Partition footprints [t_q - half, t_q + half] by a 1-D grid of unit cells starting at `origin`.

    t_q, half: [B, Q] in LR-pixel units. Returns cell indices [B, Q, K] (clamped to [0, n-1]),
    piece lengths L [B, Q, K] (zero for non-overlapping pieces) and piece midpoints [B, Q, K].
    """
    lo, hi = t_q - half, t_q + half
    first = torch.floor(lo - origin)
    K = int(torch.ceil(2 * half.max()).item()) + 1
    k = torch.arange(K, device=t_q.device, dtype=t_q.dtype)
    cell = first[..., None] + k                              # cell index (unclamped) along the axis
    a = torch.maximum(lo[..., None], cell + origin)
    b = torch.minimum(hi[..., None], cell + origin + 1)
    L = (b - a).clamp_min(0)
    mid = 0.5 * (a + b)
    idx = cell.long().clamp(0, n - 1)
    return idx, L, mid


@register('sc_inr_exact')
class SCINRExact(nn.Module):

    def __init__(self, encoder_spec, num_freqs=128, num_angles=8, freq_min=0.1, freq_max=2.0,
                 omega_bound=2.1, coef_hidden=(), learn_phase=True, base='bilinear', out_dim=3, **unused):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        d = self.encoder.out_dim
        self.K = num_freqs
        self.omega_bound = float(omega_bound)
        self.base = base
        layers, last = [], d
        for h in coef_hidden:
            layers += [nn.Conv2d(last, h, 1), nn.ReLU(inplace=True)]
            last = h
        layers.append(nn.Conv2d(last, 2 * num_freqs, 3, padding=1))
        self.coef = nn.Sequential(*layers)
        self.omega = nn.Conv2d(d, 2 * num_freqs, 3, padding=1)
        ref = init_log_polar_freqs(num_freqs, num_angles, freq_min, freq_max).flatten()
        assert float(ref.abs().max()) < self.omega_bound
        with torch.no_grad():
            nn.init.zeros_(self.omega.weight)
            self.omega.bias.copy_(torch.atanh((ref / self.omega_bound).clamp(-1 + 1e-6, 1 - 1e-6)))
        self.phase = nn.Conv2d(d, num_freqs, 1, bias=False) if learn_phase else None
        if self.phase is not None:
            nn.init.zeros_(self.phase.weight)
        self.readout = nn.Linear(2 * num_freqs, out_dim)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        self.coef_map = self.coef(self.feat)
        self.omega_map = self.omega_bound * torch.tanh(self.omega(self.feat))
        self.phase_map = self.phase(self.feat) if self.phase is not None else None
        return self.feat

    def query_rgb(self, coord, cell):
        B, Q = coord.shape[:2]
        fh, fw = self.feat.shape[-2:]
        # query centre and footprint half-width in LR-pixel units (cell i covers [i, i+1))
        yq = (coord[..., 0] + 1) / 2 * fh
        xq = (coord[..., 1] + 1) / 2 * fw
        hy = cell[..., 0] * fh / 4
        hx = cell[..., 1] * fw / 4
        iy, Ly, my = _partition(yq, hy, fh, 0.0)
        ix, Lx, mx = _partition(xq, hx, fw, 0.0)
        Ky, Kx = iy.shape[-1], ix.shape[-1]
        # all piece combinations: [B, Q, Ky*Kx]
        flat = (iy[..., :, None] * fw + ix[..., None, :]).reshape(B, Q, Ky * Kx)
        w = (Ly[..., :, None] * Lx[..., None, :]).reshape(B, Q, Ky * Kx) / (4 * hy * hx)[..., None]
        dy = (my - (iy.to(my.dtype) + 0.5))[..., :, None].expand(B, Q, Ky, Kx).reshape(B, Q, Ky * Kx)
        dx = (mx - (ix.to(mx.dtype) + 0.5))[..., None, :].expand(B, Q, Ky, Kx).reshape(B, Q, Ky * Kx)
        ly = Ly[..., :, None].expand(B, Q, Ky, Kx).reshape(B, Q, Ky * Kx)
        lx = Lx[..., None, :].expand(B, Q, Ky, Kx).reshape(B, Q, Ky * Kx)

        def gather(m):
            C = m.shape[1]
            g = m.reshape(B, C, fh * fw).gather(2, flat.reshape(B, 1, -1).expand(B, C, -1))
            return g.reshape(B, C, Q, Ky * Kx).permute(0, 2, 3, 1)      # [B, Q, P, C]

        coef = gather(self.coef_map)
        om = gather(self.omega_map).reshape(B, Q, Ky * Kx, self.K, 2)
        wy, wx = om[..., 0], om[..., 1]                                   # cycles per half LR pixel (LTE units)
        theta = 2 * math.pi * (wy * dy[..., None] + wx * dx[..., None])  # 2*pi*w*(t - t_cell) with t in LR pixels
        if self.phase_map is not None:
            theta = theta + gather(self.phase_map)
        att = torch.sinc(wy * ly[..., None]) * torch.sinc(wx * lx[..., None])   # sin(pi w L)/(pi w L): exact mean of the sinusoid over a piece of length L
        feats = torch.cat([att * torch.cos(theta), att * torch.sin(theta)], dim=-1) * coef
        out = (self.readout(feats) * w[..., None]).sum(dim=2)            # linear readout -> exact footprint mean
        if self.base == 'bilinear':
            out = out + self._base(yq, xq, hy, hx, fh, fw)
        return out

    def _base(self, yq, xq, hy, hx, fh, fw):
        """Exact footprint mean of the bilinear interpolant of the LR image (dual grid, origin at pixel centres)."""
        B, Q = yq.shape
        iy, Ly, my = _partition(yq, hy, fh - 1, 0.5)
        ix, Lx, mx = _partition(xq, hx, fw - 1, 0.5)
        Ky, Kx = iy.shape[-1], ix.shape[-1]
        w = (Ly[..., :, None] * Lx[..., None, :]).reshape(B, Q, Ky * Kx) / (4 * hy * hx)[..., None]
        gy = (my[..., :, None].expand(B, Q, Ky, Kx).reshape(B, Q, -1) / fh) * 2 - 1
        gx = (mx[..., None, :].expand(B, Q, Ky, Kx).reshape(B, Q, -1) / fw) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).reshape(B, Q * Ky * Kx, 1, 2)
        v = F.grid_sample(self.inp, grid, mode='bilinear', padding_mode='border', align_corners=False)
        v = v[:, :, :, 0].permute(0, 2, 1).reshape(B, Q, Ky * Kx, -1)
        return (v * w[..., None]).sum(dim=2)

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
