# L1 scale-collapse inference (prereg 预注册-L1L2-20261005.md). Blueprint: claude-20261005-e3/e3_h6c.py (h6c).
# --group inr  : liif/lte/sc-inr seed1 (+seed2/3 descriptive) + lmmse_generic (E3 code), pilot env, system python3
# --group gsasr / contsr (separate processes; their repos' utils/models modules collide): via claude-20260920/main_scan_gauss.py builders (scale / output size only)
# --group thera: Thera via claude-20260920/thera_main.py process() (output size only), run with thera venv312 (CPU JAX)
# Output per (group, dataset): S[img] = per-k HR energy; E[model][scale][img] = per-k error energy, k = Chebyshev nu*240.
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from common import HR, images, degrade, spec_S, spec_E, grey, CK
ap = argparse.ArgumentParser()
ap.add_argument('--group', required=True, choices=['inr', 'gsasr', 'contsr', 'thera'])
ap.add_argument('--sets', nargs='+', default=['Set14', 'BSD100', 'Urban100'])
ap.add_argument('--scales', nargs='+', type=int, default=[2, 3, 4, 6, 8, 12])
ap.add_argument('--limit', type=int, default=30)
ap.add_argument('--outdir', required=True)
a = ap.parse_args()
OUT = os.path.abspath(a.outdir); os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if (a.group != 'thera' and torch.cuda.is_available()) else 'cpu'

if a.group == 'inr':
    ROOT = '/workspace/SE-INR/scale-equivariant-inr-pilot'; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/src'); os.chdir(ROOT)
    import models, utils
    L = 512
    # ---- E3 linear code (verbatim logic, HR=240) ----
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
    @torch.no_grad()
    def render(model, lr):
        inp = (lr - 0.5) / 0.5; coord = utils.make_coord((HR, HR)).unsqueeze(0).to(dev)
        cell = torch.ones_like(coord); cell[:, :, 0] *= 2.0 / HR; cell[:, :, 1] *= 2.0 / HR
        model.gen_feat(inp); out = model.query_rgb(coord, cell)
        return (out[0] * 0.5 + 0.5).view(HR, HR, 3).permute(2, 0, 1).clamp(0, 1)
    def load(p): return models.make(torch.load(p, map_location='cpu', weights_only=False)['model'], load_sd=True).to(dev).eval()
    NETS = {}
    for mn in ['liif', 'lte', 'sc-inr']:
        NETS[mn] = load(CK / 'seed1' / mn / 'epoch-last.pth')
        for sd in (2, 3): NETS[f'{mn}@s{sd}'] = load(CK / f'seed{sd}' / mn / 'epoch-last.pth')
    def run_models(hr, lr, s):
        est = {mn: render(m, lr.to(dev)) for mn, m in NETS.items()}
        z = hr[0].mean(0).to(dev); ylr = lr[0].mean(0).to(dev)
        A = lambda zz: F.interpolate(zz[None, None, :HR, :HR], size=(HR // s, HR // s), mode='bicubic', antialias=True, align_corners=False)[0, 0]
        mu = float(ylr.mean()); var = float(z.var())
        est['lmmse_generic'] = gp_map_gen(A, ylr - mu, generic_P(2.0, var), 1e-5)[:HR, :HR] + mu
        est['bicubic'] = F.interpolate(lr, size=(HR, HR), mode='bicubic', align_corners=False)[0]
        return est
elif a.group in ('gsasr', 'contsr'):
    sys.path.insert(0, '/workspace/SE-INR/claude-20260920')
    import main_scan_gauss as MSG
    FN = {a.group: (MSG.build_gsasr if a.group == 'gsasr' else MSG.build_contsr)(dev)}
    def run_models(hr, lr, s):
        return {k: f(lr[0], float(s), HR, HR).clamp(0, 1) for k, f in FN.items()}   # 3xHRxHR
else:
    import pickle, jax, jax.numpy as jnp
    REPO = '/workspace/SE-INR/claude-20260919/thera/repo'; sys.path.insert(0, REPO); sys.path.insert(0, '/workspace/SE-INR/claude-20260920'); os.chdir(REPO)
    import thera_main as TM
    from jax import jit
    ck = pickle.load(open('/workspace/SE-INR/claude-20260919/thera/thera-edsr-pro.pkl', 'rb'))
    model = TM.build_thera(3, ck['backbone'], ck['size']); enc, dec = jit(model.apply_encoder), jit(model.apply_decoder); params = ck['model']
    def run_models(hr, lr, s):
        src = jnp.asarray(lr[0].permute(1, 2, 0).numpy(), dtype=jnp.float32)
        o = TM.process(src, enc, dec, params, (HR, HR))
        return {'thera': torch.from_numpy(np.ascontiguousarray(o.transpose(2, 0, 1))).float().clamp(0, 1)}

for ds in a.sets:
    res = dict(meta=dict(group=a.group, scales=a.scales, HR=HR, limit=a.limit, dev=dev), S={}, E={})
    t0 = time.time()
    for name, hr in images(ds, a.limit):
        z = grey(hr); res['S'][name] = spec_S(z).tolist()
        for s in a.scales:
            lr = degrade(hr, s)
            for mn, x in run_models(hr, lr, s).items():
                xg = grey(x) if x.dim() >= 3 else x.detach().double().cpu().numpy()
                assert xg.shape == (HR, HR), (mn, s, xg.shape)
                res['E'].setdefault(mn, {}).setdefault(str(s), {})[name] = spec_E(z, xg).tolist()
        print(ds, name, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(res, open(f'{OUT}/l1_{a.group}_{ds}.json', 'w'))
    print(ds, 'done', len(res['S']), 'images', flush=True)
