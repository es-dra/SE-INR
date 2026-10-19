# F1 copy of claude-20261005-e3/e3_h6c.py: models -> lte, lte-nocellphase; linear estimators skipped. prereg 预注册-F1-读出算子价值与三区-20261005.md
# E3 copy of claude-20260926/h6c.py, only change: --scale arg (S); prereg 预注册-E3-单图多倍率-冻结-20261005.md
# Put trained ASISR networks on the same axis as the a-priori linear budget: per-band recovered fraction of HR energy.
import os, sys, json, math, argparse, glob
from pathlib import Path
ROOT = Path('/workspace/SE-INR/scale-equivariant-inr-pilot'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); os.chdir(ROOT)
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import models, utils
dev = 'cuda'; HR = 256; L = 512
import argparse as _ap; _p = _ap.ArgumentParser(add_help=False); _p.add_argument('--scale', type=int, default=4); S = _p.parse_known_args()[0].scale
CK = Path('/workspace/SE-INR/SC-ASISR/artifacts/checkpoints/seed1')
def Chalf(u, sP): return torch.fft.ifft2(torch.fft.fft2(u) * sP).real
def gp_map_gen(A, y, Pspec, nv, iters=800):
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
def generic_P(alpha, var):
    f = torch.fft.fftfreq(L, device=dev); K = torch.sqrt(f[:, None] ** 2 + f[None] ** 2); P = (K + 1.0 / L) ** (-alpha); P[0, 0] = 0
    return P * var * L * L / P.sum() + 1e-12
def oracle_P(z):
    Z = torch.zeros(L, L, device=dev); Z[:HR, :HR] = z - z.mean(); return torch.abs(torch.fft.fft2(Z)) ** 2 / (HR * HR) + 1e-12
A = lambda z: F.interpolate(z[None, None, :HR, :HR], size=(HR // S, HR // S), mode='bicubic', antialias=True, align_corners=False)[0, 0]
f = torch.fft.fftfreq(HR, device=dev); mm = torch.maximum(f[:, None].abs(), f[None].abs())
BANDS = [('b1', 0, 1 / 8), ('b2', 1 / 8, 1 / 4), ('b3', 1 / 4, 0.5001)]
def bandR(x, xh):
    X = torch.fft.fft2(x - x.mean()); E = torch.fft.fft2(x - xh); ok = mm > 0
    return {n: (float((E.abs() ** 2)[(mm >= lo) & (mm < hi) & ok].sum()), float((X.abs() ** 2)[(mm >= lo) & (mm < hi) & ok].sum())) for n, lo, hi in BANDS}
@torch.no_grad()
def render(model, lr):
    inp = (lr - 0.5) / 0.5; coord = utils.make_coord((HR, HR)).unsqueeze(0).to(dev)
    cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / HR; cell[:, :, 1] *= 2.0 / HR
    model.gen_feat(inp); out = model.query_rgb(coord, cell)
    return (out[0] * 0.5 + 0.5).view(HR, HR, 3).permute(2, 0, 1).clamp(0, 1)
ap = argparse.ArgumentParser(); ap.add_argument('--sets', nargs='+', default=['Set14', 'BSD100', 'Urban100']); ap.add_argument('--limit', type=int, default=30); ap.add_argument('--out', required=True); ap.add_argument('--scale', type=int, default=4)
a = ap.parse_args()
M = {mn: models.make(torch.load(CK / mn / 'epoch-last.pth', map_location='cpu', weights_only=False)['model'], load_sd=True).to(dev).eval() for mn in ['lte', 'lte-nocellphase']}
acc = {}; R = []
for ds in a.sets:
    for fp in sorted((Path('/workspace/SE-INR/Data') / ds / 'HR').glob('*.png'))[:a.limit]:
        im = Image.open(fp).convert('RGB'); w, h = im.size
        if min(w, h) < HR: continue
        x0, y0 = (w - HR) // 2, (h - HR) // 2
        hr = transforms.ToTensor()(im.crop((x0, y0, x0 + HR, y0 + HR)))[None].to(dev)
        lr = F.interpolate(hr, size=(HR // S, HR // S), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
        z = hr[0].mean(0); ylr = lr[0].mean(0)
        est = {'bicubic': F.interpolate(lr, size=(HR, HR), mode='bicubic', align_corners=False)[0].mean(0)}
        for mn, m in M.items(): est[mn] = render(m, lr).mean(0)
        mu = float(ylr.mean()); var = float(z.var())
        pass  # F1: linear skipped
        #est['lmmse_generic'] = gp_map_gen(A, ylr - mu, generic_P(2.0, var), 1e-5)[:HR, :HR] + mu
        #est['lmmse_oracle'] = gp_map_gen(A, ylr - mu, oracle_P(z), 1e-5)[:HR, :HR] + mu
        rec = dict(set=ds, img=fp.name)
        for k, v in est.items():
            br = bandR(z, v)
            for b, (e, s) in br.items():
                acc.setdefault((ds, k, b), [0, 0]); acc[(ds, k, b)][0] += e; acc[(ds, k, b)][1] += s; rec[f'{k}_{b}'] = 1 - e / s
        R.append(rec)
    json.dump(dict(per_image=R, pooled={f'{d}|{k}|{b}': 1 - v[0] / v[1] for (d, k, b), v in acc.items()}), open(a.out, 'w'))
    print(ds, 'done', flush=True)
for (d, k, b), v in sorted(acc.items()): print(d, k, b, round(1 - v[0] / v[1], 3))
