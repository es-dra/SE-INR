# sparse-pixel sampling-rate scan on the budget axis: per-band recovered fraction for linear (generic / oracle LMMSE) vs DIP (nonlinear, untrained)
import sys, os, json, time, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/SE-INR/claude-20260924/sp2d')
from common import load_set, mask_for
from dip import dip_inpaint
dev = 'cuda'; HR = 256; L = 512
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
f = torch.fft.fftfreq(HR, device=dev); mm = torch.maximum(f[:, None].abs(), f[None].abs())
BANDS = [('b1', 0, 1 / 8), ('b2', 1 / 8, 1 / 4), ('b3', 1 / 4, 0.5001)]
def bandE(x, xh):
    X = torch.fft.fft2(x - x.mean()); E = torch.fft.fft2(x - xh); ok = mm > 0
    return {n: (float((E.abs() ** 2)[(mm >= lo) & (mm < hi) & ok].sum()), float((X.abs() ** 2)[(mm >= lo) & (mm < hi) & ok].sum())) for n, lo, hi in BANDS}
ap = argparse.ArgumentParser(); ap.add_argument('--rhos', nargs='+', type=float, default=[0.05, 0.1, 0.25, 0.5]); ap.add_argument('--n', type=int, default=12); ap.add_argument('--dip', type=int, default=1); ap.add_argument('--out', required=True)
a = ap.parse_args(); acc = {}; R = []; t0 = time.time()
for name, img in load_set('div2k')[:a.n]:
    x = torch.tensor(img, device=dev).permute(2, 0, 1); z = x.mean(0)
    for rho in a.rhos:
        m = torch.tensor(mask_for(name, rho), device=dev)
        A = lambda q: q[:HR, :HR][m]
        y = z[m]; mu = float(y.mean()); var = float(y.var())
        est = {'lmmse_generic': gp_map_gen(A, y - mu, generic_P(2.0, var), 1e-5)[:HR, :HR] + mu,
               'lmmse_oracle': gp_map_gen(A, y - mu, oracle_P(z), 1e-5)[:HR, :HR] + mu}
        if a.dip: est['dip'] = torch.tensor(dip_inpaint(img, mask_for(name, rho), iters=(3000,))[3000], device=dev).mean(-1)
        rec = dict(img=name, rho=rho)
        for k, v in est.items():
            rec[f'{k}_psnr'] = float(-10 * torch.log10(((v.clamp(0, 1) - z)[8:-8, 8:-8] ** 2).mean()))
            for b, (e, s) in bandE(z, v).items():
                acc.setdefault((rho, k, b), [0, 0]); acc[(rho, k, b)][0] += e; acc[(rho, k, b)][1] += s
        R.append(rec)
        json.dump(dict(per=R, pooled={f'{r}|{k}|{b}': 1 - v[0] / v[1] for (r, k, b), v in acc.items()}), open(a.out, 'w'))
    print(name, f'{time.time()-t0:.0f}s', flush=True)
for (r, k, b), v in sorted(acc.items()): print(r, k, b, round(1 - v[0] / v[1], 3))
for rho in a.rhos:
    for k in ('lmmse_generic', 'lmmse_oracle', 'dip'):
        vals = [q[f'{k}_psnr'] for q in R if q['rho'] == rho and f'{k}_psnr' in q]
        if vals: print('PSNR', rho, k, round(float(np.mean(vals)), 2))
