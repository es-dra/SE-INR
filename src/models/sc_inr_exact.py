"""SC-INR-exact: a continuous local Fourier field whose output is the exact mean over the query footprint.

Field. Each LR feature cell j (centre t_j = j + 1/2 in LR-pixel units) carries a local expansion
    f_j(y, x) = sum_k a_jk cos(2*pi*(w_jky (y - y_j) + w_jkx (x - x_j)) + phi_jk).
The continuous field blends the four nearest cells with bilinear (tent) weights on the dual grid,
    f(y, x) = sum_j w_j(y, x) f_j(y, x),   w_j = (1 - |y - y_j|)(1 - |x - x_j|)_+ ,
so it is continuous everywhere. In the limit of a vanishing footprint this is exactly the area-weighted
local ensemble of LIIF/LTE: the ensemble is the zero-footprint case of this model.

Output. A query with centre (y_q, x_q) and footprint (c_y, c_x) returns the exact mean of f over its
footprint rectangle. The rectangle is partitioned by the dual grid; on each piece the weights are
bilinear and the sinusoid integrates in closed form through two real 1-D kernels per axis,
    S(A, L) = int_{-L/2}^{L/2} e^{iAt} dt = L sinc(AL/2),
    T(A, L) = int_{-L/2}^{L/2} t e^{iAt} dt = -i (L^2/2) sinc'(AL/2),
so that for w = (a + b t_y)(c + d t_x) in piece-centred coordinates
    int w cos(A t_y + B t_x + theta) = Re[(aS_y + bT_y)(cS_x + dT_x) e^{i theta}].
Because the readout is linear in these features, the output is exactly the footprint mean: an m x m box
average of the s-scale output equals the s/m-scale output, asymmetric footprints, downscaling (c > 1) and
out-of-range cells all use the same formula, and the field has no seams at large scales.

The low-frequency base term is the exact footprint mean of the bilinear interpolant of the LR image.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models.sc_inr_adaptive import init_log_polar_freqs


def _dsinc(z):
    """d/dz [sin z / z], stable near zero (series below |z| < 1e-2)."""
    small = z.abs() < 1e-2
    zs = torch.where(small, torch.ones_like(z), z)
    exact = (zs * torch.cos(zs) - torch.sin(zs)) / (zs * zs)
    series = -z / 3 + z ** 3 / 30 - z ** 5 / 840
    return torch.where(small, series, exact)


def _pieces(t_q, half, n_dual):
    """Partition footprints [t_q - half, t_q + half] by the dual grid (cell centres at i + 1/2) along one axis.

    Returns, per piece: dual index p (unclamped, piece lies between centres p+1/2 and p+3/2), piece length L (0 if
    empty), piece midpoint mid, and the two neighbour cell indices (clamped) with their tent weights in
    piece-centred coordinates, w0 = a0 + b0*t and w1 = a1 + b1*t. Outside the dual grid the nearest cell gets
    weight one (border replication). Shapes [B, Q, K].
    """
    lo, hi = t_q - half, t_q + half
    first = torch.floor(lo - 0.5)
    K = int(torch.ceil(2 * half.max()).item()) + 1
    k = torch.arange(K, device=t_q.device, dtype=t_q.dtype)
    p = first[..., None] + k
    a = torch.maximum(lo[..., None], p + 0.5)
    b = torch.minimum(hi[..., None], p + 1.5)
    L = (b - a).clamp_min(0)
    mid = 0.5 * (a + b)
    inside = (p >= 0) & (p <= n_dual - 1)
    below, above = p < 0, p > n_dual - 1
    j0 = p.clamp(0, n_dual).long()
    j1 = (p + 1).clamp(0, n_dual).long()
    a0 = torch.where(inside, (p + 1.5) - mid, torch.where(below, torch.ones_like(mid), torch.zeros_like(mid)))
    b0 = torch.where(inside, -torch.ones_like(mid), torch.zeros_like(mid))
    a1 = torch.where(inside, mid - (p + 0.5), torch.where(above, torch.ones_like(mid), torch.zeros_like(mid)))
    b1 = torch.where(inside, torch.ones_like(mid), torch.zeros_like(mid))
    return L, mid, (j0, a0, b0), (j1, a1, b1)


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
        yq = (coord[..., 0] + 1) / 2 * fh
        xq = (coord[..., 1] + 1) / 2 * fw
        hy = cell[..., 0] * fh / 4
        hx = cell[..., 1] * fw / 4
        area = (4 * hy * hx)[..., None]                                   # [B, Q, 1]
        Ly, my, (jy0, ay0, by0), (jy1, ay1, by1) = _pieces(yq, hy, fh - 1)
        Lx, mx, (jx0, ax0, bx0), (jx1, ax1, bx1) = _pieces(xq, hx, fw - 1)
        Ky, Kx = Ly.shape[-1], Lx.shape[-1]
        P = Ky * Kx * 4

        def grid(y, x):                                                   # [B,Q,Ky] x [B,Q,Kx] -> [B,Q,Ky*Kx]
            return (y[..., :, None] * x[..., None, :]).reshape(B, Q, Ky * Kx)

        def tile(v):                                                      # [B,Q,Ky*Kx] -> [B,Q,P]
            return v.repeat_interleave(4, dim=-1)

        ones_y, ones_x = torch.ones_like(Ly), torch.ones_like(Lx)
        L2 = tile(grid(Ly, Lx))                                           # piece areas
        my_t, mx_t = tile(grid(my, ones_x)), tile(grid(ones_y, mx))
        Ly_t, Lx_t = tile(grid(Ly, ones_x)), tile(grid(ones_y, Lx))
        # neighbour order within a piece: (y0,x0), (y0,x1), (y1,x0), (y1,x1)
        def pick(v0, v1, axis):
            g0, g1 = (grid(v0, ones_x), grid(v1, ones_x)) if axis == 'y' else (grid(ones_y, v0), grid(ones_y, v1))
            order = (g0, g0, g1, g1) if axis == 'y' else (g0, g1, g0, g1)
            return torch.stack(order, dim=-1).reshape(B, Q, P)
        jy = pick(jy0.to(Ly.dtype), jy1.to(Ly.dtype), 'y').long()
        jx = pick(jx0.to(Lx.dtype), jx1.to(Lx.dtype), 'x').long()
        ay, by = pick(ay0, ay1, 'y'), pick(by0, by1, 'y')
        ax, bx = pick(ax0, ax1, 'x'), pick(bx0, bx1, 'x')
        flat = jy * fw + jx                                               # [B, Q, P]

        def gather(m):
            C = m.shape[1]
            g = m.reshape(B, C, fh * fw).gather(2, flat.reshape(B, 1, -1).expand(B, C, -1))
            return g.reshape(B, C, Q, P).permute(0, 2, 3, 1)             # [B, Q, P, C]

        coef = gather(self.coef_map)
        om = gather(self.omega_map).reshape(B, Q, P, self.K, 2)
        wy, wx = om[..., 0], om[..., 1]                                   # cycles per LR pixel
        dy = my_t - (jy.to(my_t.dtype) + 0.5)
        dx = mx_t - (jx.to(mx_t.dtype) + 0.5)
        theta = 2 * math.pi * (wy * dy[..., None] + wx * dx[..., None])
        if self.phase_map is not None:
            theta = theta + gather(self.phase_map)
        zy, zx = math.pi * wy * Ly_t[..., None], math.pi * wx * Lx_t[..., None]
        Sy, Sx = Ly_t[..., None] * torch.sinc(wy * Ly_t[..., None]), Lx_t[..., None] * torch.sinc(wx * Lx_t[..., None])
        Py, Px = ay[..., None] * Sy, ax[..., None] * Sx
        Qy = -by[..., None] * (Ly_t[..., None] ** 2 / 2) * _dsinc(zy)
        Qx = -bx[..., None] * (Lx_t[..., None] ** 2 / 2) * _dsinc(zx)
        Rr = Py * Px - Qy * Qx
        Ii = Py * Qx + Qy * Px
        c, s = torch.cos(theta), torch.sin(theta)
        feats = torch.cat([Rr * c - Ii * s, Rr * s + Ii * c], dim=-1) * coef   # [B, Q, P, 2K], already area-weighted
        out = F.linear(feats / area[..., None], self.readout.weight).sum(dim=2) + self.readout.bias
        if self.base == 'bilinear':
            out = out + self._base(yq, xq, hy, hx, fh, fw, L2, my_t, mx_t, area)
        return out

    def _base(self, yq, xq, hy, hx, fh, fw, L2, my_t, mx_t, area):
        """Exact footprint mean of the bilinear LR interpolant: bilinear on each dual-grid piece, so its mean is its midpoint value."""
        B, Q = yq.shape
        P = L2.shape[-1]
        w = (L2 / area)[..., ::4]                                          # one weight per piece (the 4 neighbour copies share it)
        gy = (my_t[..., ::4] / fh) * 2 - 1
        gx = (mx_t[..., ::4] / fw) * 2 - 1
        g = torch.stack([gx, gy], dim=-1).reshape(B, -1, 1, 2)
        v = F.grid_sample(self.inp, g, mode='bilinear', padding_mode='border', align_corners=False)
        v = v[:, :, :, 0].permute(0, 2, 1).reshape(B, Q, P // 4, -1)
        return (v * w[..., None]).sum(dim=2)

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
