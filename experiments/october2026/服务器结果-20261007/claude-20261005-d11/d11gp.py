# D11: built from claude-20261004-e1/e1gp.py by mk_d11.py (cpu + band energies); prereg 预注册-D11-E1切换提前诊断-20261005.md
# E1 copy of d10 (claude-20260928), only change: --T (and --skip for gp) args; prereg 预注册-E2-E1-谱失配代价与多帧全扫-20261003
# D3: multi-frame continuous SR feasibility (preregistered: claude/预注册-D3-多帧连续超分可行性-20260928.md)
import os, sys, json, math, time, argparse, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
dev = 'cpu'; S = 4; HR = 256; LR = HR // S; T = 16
NAT = {}
def load(n):
    return load_native(n)
def load_old(n):
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
def load_native(n):
    fs = sorted(glob.glob('/workspace/SE-INR/Data/DIV2K_valid_HR/*.png'))[:n]; out = []
    for f in fs:
        im = Image.open(f).convert('RGB'); w, h = im.size; x0, y0 = (w - 2 * HR) // 2, (h - 2 * HR) // 2
        nat = torch.tensor(np.asarray(im.crop((x0, y0, x0 + 2 * HR, y0 + 2 * HR)), dtype=np.float32) / 255.).permute(2, 0, 1)
        nm = os.path.basename(f)[:-4]; NAT[nm] = nat; out.append((nm, F.avg_pool2d(nat[None], 2)[0]))
    return out
def nat_eval(nat, pts):
    g = (pts * 2 / (2 * HR) * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(nat[None], g, mode='bicubic', padding_mode='border', align_corners=False)[0, :, 0].T
def hr_eval(img, pts):  # bicubic continuous HR at pixel coords pts (N,2)
    g = (pts / HR * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(img[None], g, mode='bicubic', padding_mode='border', align_corners=False)[0, :, 0].T  # (N,3)
def make_obs(img, name, noise):
    d, a = frame_params(name, T); P = sample_points(d, a, 16).to(dev); img = img.to(dev); Y = []
    nat = NAT[name].to(dev)
    for t in range(T): Y.append(nat_eval(nat, P[t]).view(LR * LR, 256, 3).mean(1))
    Y = torch.stack(Y)
    if noise > 0: Y = Y + noise * torch.randn(Y.shape, generator=torch.Generator(device=dev).manual_seed(1), device=dev)
    return Y, d, a  # (T, LR*LR, 3)

L = 512; KS = 8
def A_op(x, P):  # x (3,L,L) torus field; reads HR region with bicubic at KS x KS subsamples per LR pixel, box average
    g = (P.view(1, 1, -1, 2) * 2 - 1)
    v = F.grid_sample(x[None, :, :HR, :HR], g, mode='bicubic', padding_mode='border', align_corners=False)[0, :, 0].T
    return v.view(T, LR * LR, KS * KS, 3).mean(2)
def At_op(y, P):
    z = torch.zeros(3, L, L, device=dev, requires_grad=True)
    with torch.enable_grad(): g, = torch.autograd.grad(A_op(z, P), z, grad_outputs=y)
    return g
def Chalf(u, sP): return torch.fft.ifft2(torch.fft.fft2(u) * sP).real
def per_psd(x, smooth=0):
    z = x.mean(0); z = z - z.mean(); Z = torch.zeros(L, L, device=dev); Z[:HR, :HR] = z
    Pp = torch.abs(torch.fft.fft2(Z)) ** 2 / (HR * HR)
    if smooth:
        f = torch.fft.fftfreq(L, device=dev) * L; gg = torch.exp(-0.5 * (f / smooth) ** 2); G = gg[:, None] * gg[None]; G = G / G.sum()
        Pp = torch.fft.ifft2(torch.fft.fft2(Pp) * torch.fft.fft2(G)).real.clamp(min=0)
    return Pp + 1e-8 * float(Pp.mean())
def gp_map(Y, P, Pspec, nv, iters=600):
    """posterior mean, preconditioned: x = mu + C^1/2 u,  (C^1/2 A^T A C^1/2 / nv + I) u = C^1/2 A^T (y-mu) / nv"""
    sP = Pspec.sqrt(); mu = Y.mean((0, 1)); b = Y - mu
    def M(u): return Chalf(At_op(A_op(Chalf(u, sP), P), P), sP) / nv + u
    rhs = Chalf(At_op(b, P), sP) / nv
    u = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone(); rs = (r * r).sum()
    for _ in range(iters):
        Mp = M(p); al = rs / (p * Mp).sum(); u = u + al * p; r = r - al * Mp; rn = (r * r).sum()
        if rn.sqrt() < 1e-5 * rhs.norm(): break
        p = r + rn / rs * p; rs = rn
    return (Chalf(u, sP)[:, :HR, :HR] + mu[:, None, None]).clamp(0, 1)
def psnr(x, y, sh=8): return float(-10 * torch.log10(((x - y.to(x.device))[:, sh:-sh, sh:-sh] ** 2).mean()))
def gauss_P(sig, var):
    f = torch.fft.fftfreq(L, device=dev) * L / 2.0  # cycles per image (image = HR px)
    K2 = f[:, None] ** 2 + f[None] ** 2; P = torch.exp(-K2 / (2 * sig ** 2)); P[0, 0] = 0
    return P * var * L * L / P.sum() + 1e-10
def power_P(alpha, var):
    f = torch.fft.fftfreq(L, device=dev); K = torch.sqrt(f[:, None] ** 2 + f[None] ** 2); P = (K + 1.0 / L) ** (-alpha); P[0, 0] = 0
    return P * var * L * L / P.sum() + 1e-10

import os
torch.set_num_threads(int(os.environ.get("NTH", "8")))
def bands_HR(n):  # cycles/px on n-grid
    f = torch.fft.fftfreq(n); m = torch.maximum(f[:, None].abs(), f[None].abs())
    return [(m < 1 / 8), (m >= 1 / 8) & (m < 1 / 4), (m >= 1 / 4)]
BW = bands_HR(240); BL = [b.to(dev) for b in bands_HR(L)]
hann = torch.hann_window(240, periodic=False); W2 = hann[:, None] * hann[None]
def band_energy(z):  # z (3,256,256) -> per band windowed energy summed over channels
    c = z[:, 8:-8, 8:-8] * W2; S = (torch.fft.fft2(c).abs() ** 2).sum(0)
    return [float(S[b].sum()) for b in BW]
def band_P(P): return [float(P[b].sum()) for b in BL]
import numpy as np
def spec(z): c = z[:, 8:-8, 8:-8] * W2; return (torch.fft.fft2(c).abs() ** 2).sum(0).numpy().astype(np.float32)
ap = argparse.ArgumentParser(); ap.add_argument('--skip', type=int, default=0); ap.add_argument('--n', type=int, default=1); ap.add_argument('--noise', type=float, default=0.005); ap.add_argument('--out', required=True); ap.add_argument('--T', type=int, default=16)
a = ap.parse_args(); T = a.T; R = []; t0 = time.time()
for name, img in load(12)[a.skip:a.skip + a.n]:
    torch.manual_seed(0); Y, d, fa = make_obs(img, name, a.noise); P = sample_points(d, fa, KS).to(dev) / HR
    r = dict(img=name, noise=a.noise, T=T, E=band_energy(img), arms={}); imgd = img.to(dev); nv = max(a.noise ** 2, 1e-5); var = float(Y.var()) * 1.0
    specs = {f'gp_gauss_{s}': gauss_P(s, var) for s in (3, 10, 30, 60, 128)}
    specs.update({f'gp_power_{al}': power_P(al, var) for al in (2.0, 2.5, 3.0)}); specs['gp_oracle'] = per_psd(imgd)
    SP = {'truth': spec(imgd)}
    for k, Ps in specs.items():
        xh = gp_map(Y, P, Ps, nv)
        r['arms'][k] = dict(psnr=psnr(xh, imgd), err=band_energy(xh - imgd), Pb=band_P(Ps)); SP[k] = spec(xh - imgd)
        print(name, a.noise, k, round(r['arms'][k]['psnr'], 2), f'{time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(a.out.replace('.json', f'_{name}_spec.npz'), **SP)
    R.append(r); json.dump(R, open(a.out, 'w')); print(name, a.noise, {k: round(v['psnr'], 2) for k, v in r['arms'].items()}, f'{time.time()-t0:.0f}s', flush=True)
print('done')
