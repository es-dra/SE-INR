# E1 copy of d10 (claude-20260928), only change: --T (and --skip for gp) args; prereg 预注册-E2-E1-谱失配代价与多帧全扫-20261003
# D3: multi-frame continuous SR feasibility (preregistered: claude/预注册-D3-多帧连续超分可行性-20260928.md)
import os, sys, json, math, time, argparse, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
dev = 'cuda'; S = 4; HR = 256; LR = HR // S; T = 16
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
# ---------------- representations: f(v) with v in [0,1)^2 (x,y) -> rgb ----------------
class FF(nn.Module):
    def __init__(s, B, widths=(256, 256, 256, 256)):
        super().__init__(); s.register_buffer('B', B); L = []; i = 2 * B.shape[0]
        for w in widths: L += [nn.Linear(i, w), nn.ReLU()]; i = w
        s.net = nn.Sequential(*L, nn.Linear(i, 3))
    def forward(s, v): z = 2 * math.pi * v @ s.B.T; return s.net(torch.cat([torch.cos(z), torch.sin(z)], -1))
def gauss_B(sig, m=256, seed=0): return torch.randn(m, 2, generator=torch.Generator().manual_seed(seed)) * sig
def ring_B(lo, hi, m, seed):
    g = torch.Generator().manual_seed(seed); r = lo + (hi - lo) * torch.rand(m, generator=g); th = 2 * math.pi * torch.rand(m, generator=g)
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], -1)
class BandFF(nn.Module):
    def __init__(s):
        super().__init__()
        s.subs = nn.ModuleList([FF(ring_B(0, 16, 64, 1), (64, 64, 64)), FF(ring_B(16, 48, 128, 2), (128, 128, 128)), FF(ring_B(48, 128, 256, 3), (256, 256, 256))])
    def forward(s, v): return sum(m(v) for m in s.subs)
class MRGrid(nn.Module):
    def __init__(s, res=(16, 32, 64, 128, 256), Fd=4):
        super().__init__(); s.G = nn.ParameterList([nn.Parameter(torch.empty(1, Fd, r, r).uniform_(-1e-4, 1e-4)) for r in res])
        s.net = nn.Sequential(nn.Linear(Fd * len(res), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(s, v):
        g = (v * 2 - 1).view(1, 1, -1, 2)
        f = torch.cat([F.grid_sample(G, g, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T for G in s.G], -1)
        return s.net(f)
class Grid(nn.Module):
    def __init__(s, init): super().__init__(); s.G = nn.Parameter(init.clone()[None])
    def forward(s, v): return F.grid_sample(s.G, (v * 2 - 1).view(1, 1, -1, 2), mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T
    def tv(s): G = s.G; return (G[..., 1:, :] - G[..., :-1, :]).abs().mean() + (G[..., :, 1:] - G[..., :, :-1]).abs().mean()
def fit(model, Y, d, a, steps=2000, lr=1e-3, tv=0.0, grid_lr=None):
    P = sample_points(d, a, 4).to(dev) / HR  # (T, LR*LR*16, 2)
    params = [{'params': [p for n, p in model.named_parameters() if not n.startswith('G')], 'lr': lr}]
    gp = [p for n, p in model.named_parameters() if n.startswith('G')]
    if gp: params.append({'params': gp, 'lr': grid_lr or lr})
    opt = torch.optim.Adam(params); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    Yf = Y.reshape(-1, 3); Pf = P.reshape(-1, 2)
    for it in range(steps):
        opt.zero_grad(); pred = model(Pf).view(T * LR * LR, 16, 3).mean(1)
        loss = ((pred - Yf) ** 2).mean() + (tv * model.tv() if tv else 0); loss.backward(); opt.step(); sch.step()
    return model
@torch.no_grad()
def render(model):
    ii, jj = torch.meshgrid(torch.arange(HR), torch.arange(HR), indexing='ij')
    v = torch.stack([(jj + .5) / HR, (ii + .5) / HR], -1).view(-1, 2).to(dev)
    return model(v).T.reshape(3, HR, HR).clamp(0, 1)
def psnr(x, y, sh=8): return float(-10 * torch.log10(((x - y.to(x.device))[:, sh:-sh, sh:-sh] ** 2).mean()))
ap = argparse.ArgumentParser(); ap.add_argument('--skip', type=int, default=0); ap.add_argument('--n', type=int, default=3); ap.add_argument('--noise', type=float, default=0.0)
ap.add_argument('--out', required=True); ap.add_argument('--T', type=int, default=16); ap.add_argument('--steps', type=int, default=2000); a = ap.parse_args(); T = a.T
R = []; t0 = time.time()
for name, img in load(12)[a.skip:a.skip + a.n]:
    torch.manual_seed(0); Y, d, fa = make_obs(img, name, a.noise); r = dict(img=name, noise=a.noise, T=T)
    ref = Y[0].T.reshape(1, 3, LR, LR); bil = F.interpolate(ref, scale_factor=S, mode='bilinear', align_corners=False)[0]
    r['bilinear'] = psnr(bil.clamp(0, 1), img)
    for sig in (3, 10):
        torch.manual_seed(0); r[f"ff_{sig}"] = psnr(render(fit(FF(gauss_B(sig)).to(dev), Y, d, fa, a.steps)), img)
    torch.manual_seed(0); r["mrgrid"] = psnr(render(fit(MRGrid().to(dev), Y, d, fa, a.steps, lr=1e-3, grid_lr=1e-2)), img)
    r["ls_tv1e-4"] = psnr(render(fit(Grid(bil).to(dev), Y, d, fa, a.steps, lr=1e-2, tv=1e-4, grid_lr=1e-2)), img)
    r["ls_tv1e-3"] = psnr(render(fit(Grid(bil).to(dev), Y, d, fa, a.steps, lr=1e-2, tv=1e-3, grid_lr=1e-2)), img)
    R.append(r); json.dump(R, open(a.out, 'w')); print(name, {k: round(v, 2) for k, v in r.items() if isinstance(v, float)}, f'{time.time()-t0:.0f}s', flush=True)
print('done')
