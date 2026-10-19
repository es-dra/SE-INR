# E3 prior budget: copy of claude-20260928/d8.py, only change: task list = single x2/x3/x4/x8 (adds x3)
# D3: multi-frame continuous SR feasibility (preregistered: claude/预注册-D3-多帧连续超分可行性-20260928.md)
import os, sys, json, math, time, argparse, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
dev = 'cuda'; S = 4; HR = 256; LR = HR // S; T = 16
def load(n):
    fs = sorted(glob.glob('/workspace/SE-INR/Data/DIV2K_valid_HR/*.png'))[:n]; out = []
    for f in fs:
        im = Image.open(f).convert('RGB'); w, h = im.size; x0, y0 = (w - HR) // 2, (h - HR) // 2
        out.append((os.path.basename(f)[:-4], torch.tensor(np.asarray(im.crop((x0, y0, x0 + HR, y0 + HR)), dtype=np.float32) / 255.).permute(2, 0, 1)))
    return out
def frame_params(name, T):
    import zlib; g = np.random.default_rng(zlib.crc32(name.encode()))
    d = g.uniform(-1, 1, (T, 2)) * S; a = g.uniform(-1, 1, T) * math.pi / 180; d[0] = 0; a[0] = 0
    return torch.tensor(d, dtype=torch.float32), torch.tensor(a, dtype=torch.float32)
def sample_points(d, a, k):
    """reference-frame HR-pixel coords of k x k subsamples of every LR pixel of every frame: (T, LR*LR*k*k, 2) as (x,y)"""
    ii, jj = torch.meshgrid(torch.arange(LR), torch.arange(LR), indexing='ij')
    c = torch.stack([(jj + .5) * S, (ii + .5) * S], -1).view(-1, 1, 2).float()
    o = (torch.arange(k) + .5) / k * S - S / 2; oy, ox = torch.meshgrid(o, o, indexing='ij'); off = torch.stack([ox, oy], -1).view(1, -1, 2)
    p = (c + off).view(-1, 2) - HR / 2; P = []
    for t in range(len(d)):
        R = torch.tensor([[math.cos(a[t]), -math.sin(a[t])], [math.sin(a[t]), math.cos(a[t])]])
        P.append(p @ R.T + HR / 2 + d[t])
    return torch.stack(P)
def hr_eval(img, pts):  # bicubic continuous HR at pixel coords pts (N,2)
    g = (pts / HR * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(img[None], g, mode='bicubic', padding_mode='border', align_corners=False)[0, :, 0].T  # (N,3)
def make_obs(img, name, noise):
    d, a = frame_params(name, T); P = sample_points(d, a, 16).to(dev); img = img.to(dev); Y = []
    for t in range(T): Y.append(hr_eval(img, P[t]).view(LR * LR, 256, 3).mean(1))
    Y = torch.stack(Y)
    if noise > 0: Y = Y + noise * torch.randn(Y.shape, generator=torch.Generator(device=dev).manual_seed(1), device=dev)
    return Y, d, a  # (T, LR*LR, 3)


L = 512
def Chalf(u, sP): return torch.fft.ifft2(torch.fft.fft2(u) * sP).real
def gp_map_gen(A, y, Pspec, nv, iters=800):
    """LMMSE / posterior mean for zero-mean GP on the L-torus (1 channel), generic linear A reading the HR window."""
    sP = Pspec.sqrt()
    def At(v):
        z = torch.zeros(L, L, device=dev, requires_grad=True)
        with torch.enable_grad(): g, = torch.autograd.grad(A(z), z, grad_outputs=v)
        return g
    def M(u): return Chalf(At(A(Chalf(u, sP))), sP) / nv + u
    rhs = Chalf(At(y), sP) / nv; u = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone(); rs = (r * r).sum()
    for _ in range(iters):
        Mp = M(p); al = rs / (p * Mp).sum(); u = u + al * p; r = r - al * Mp; rn = (r * r).sum()
        if rn.sqrt() < 1e-6 * rhs.norm(): break
        p = r + rn / rs * p; rs = rn
    return Chalf(u, sP)
def generic_P(alpha=2.0, var=0.05):
    f = torch.fft.fftfreq(L, device=dev); K = torch.sqrt(f[:, None] ** 2 + f[None] ** 2)
    P = (K + 1.0 / L) ** (-alpha); P[0, 0] = 0
    return P * var * L * L / P.sum()  # pixel variance = var
def sample_prior(P, g):
    w = torch.randn(L, L, device=dev, generator=g); return Chalf(w, P.sqrt())
BANDS = [('b1', 0, 1 / 8), ('b2', 1 / 8, 1 / 4), ('b3', 1 / 4, 0.5001)]
def band_masks():
    f = torch.fft.fftfreq(HR, device=dev); m = torch.maximum(f[:, None].abs(), f[None].abs())
    return [(n, (m >= lo) & (m < hi)) for n, lo, hi in BANDS]
def band_R(x, xh, masks):
    X = torch.fft.fft2(x[8:-8, 8:-8] if False else x); E = torch.fft.fft2(x - xh)
    return {n: float(1 - (E.abs() ** 2)[mk].sum() / (X.abs() ** 2)[mk].sum()) for n, mk in masks}
# ---------------- observation operators on the HR window of the torus field ----------------
def op_single(s, sig):
    h = HR // s
    return (lambda z: F.interpolate(z[None, None, :HR, :HR], size=(h, h), mode='bicubic', antialias=True, align_corners=False)[0, 0]), sig
def op_sparse(rho, sig, seed=0):
    m = torch.rand(HR, HR, generator=torch.Generator().manual_seed(seed)) < rho; m = m.to(dev)
    return (lambda z: z[:HR, :HR][m]), sig
def op_multi(Tn, sig, name='0801'):
    global T
    T0 = T; T = Tn; d, a = frame_params(name, Tn); P = (sample_points(d, a, 8) / HR).to(dev); T = T0
    def A(z):
        g = (P.view(1, 1, -1, 2) * 2 - 1)
        return F.grid_sample(z[None, None, :HR, :HR], g, mode='bicubic', padding_mode='border', align_corners=False)[0, 0, 0].view(Tn, LR * LR, 64).mean(2)
    return A, sig
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--nsamp', type=int, default=3); ap.add_argument('--alpha', type=float, default=2.0)
a = ap.parse_args(); P = generic_P(a.alpha); masks = band_masks(); g = torch.Generator(device=dev).manual_seed(0)
tasks = []
for s in (2, 3, 4, 8):
    for sig in (0.002, 0.01): tasks.append((f'single_x{s}_n{sig}', op_single(s, sig)))
R = []; t0 = time.time()
for name, (A, sig) in tasks:
    acc = {n: [] for n, _, _ in BANDS}
    for k in range(a.nsamp):
        x = sample_prior(P, g); y = A(x); y = y + sig * torch.randn(y.shape, device=dev, generator=g)
        xh = gp_map_gen(A, y, P, sig ** 2)
        for n, v in band_R(x[:HR, :HR], xh[:HR, :HR], masks).items(): acc[n].append(v)
    r = dict(task=name, alpha=a.alpha, **{f'R_{n}': float(np.mean(v)) for n, v in acc.items()}); R.append(r)
    json.dump(R, open(a.out, 'w')); print(name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, f'{time.time()-t0:.0f}s', flush=True)
print('done')
