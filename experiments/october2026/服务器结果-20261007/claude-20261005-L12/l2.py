# L2 readout operator = scale information (prereg 预注册-L1L2-20261005.md). Blueprint: e3_h6c.py render().
# Pairs (full, ablate): (lte, lte-nocellphase), (sc-inr, sc-inr-nosinc), seed1 epoch-last.
# f_full : full model, cell = 2/HR (= 2/(h*s)), HR output grid.
# f_abl  : ablated model, same query.
# f_marg : mean over s' in {1.0,1.1,...,4.0} (31, uniform as training U[1,4]) of clamped full-model outputs with cell = 2/(h*s'), HR grid.
# f_mlog : same, s' log-uniform in [1,4] (31 points; descriptive).
# H2z    : max |f_abl(cell) - f_abl(cell')| over cells {2/HR, s'=1, s'=4}, on unclamped outputs.
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from common import HR, images, degrade, spec_S, spec_E, grey, CK
ROOT = '/workspace/SE-INR/scale-equivariant-inr-pilot'; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/src'); os.chdir(ROOT)
import models, utils
ap = argparse.ArgumentParser()
ap.add_argument('--sets', nargs='+', default=['Set14', 'BSD100', 'Urban100'])
ap.add_argument('--scales', nargs='+', type=int, default=[2, 3, 4, 8])
ap.add_argument('--limit', type=int, default=30)
ap.add_argument('--outdir', required=True)
a = ap.parse_args()
OUT = os.path.abspath(a.outdir); os.makedirs(OUT, exist_ok=True)
dev = 'cuda'
PAIRS = [('lte', 'lte-nocellphase'), ('sc-inr', 'sc-inr-nosinc')]
SLIN = np.round(np.linspace(1.0, 4.0, 31), 6); SLOG = np.exp(np.linspace(0.0, np.log(4.0), 31))
def load(mn): return models.make(torch.load(CK / 'seed1' / mn / 'epoch-last.pth', map_location='cpu', weights_only=False)['model'], load_sd=True).to(dev).eval()
NET = {mn: load(mn) for p in PAIRS for mn in p}
coord = utils.make_coord((HR, HR)).unsqueeze(0).to(dev)
@torch.no_grad()
def query(m, c):
    cell = torch.full_like(coord, float(c))
    return (m.query_rgb(coord, cell)[0] * 0.5 + 0.5).view(HR, HR, 3).permute(2, 0, 1)   # unclamped
def mse(x, y): return float(((x - y) ** 2).mean())

for ds in a.sets:
    res = dict(meta=dict(HR=HR, scales=a.scales, limit=a.limit, slin=SLIN.tolist(), slog=SLOG.tolist()), S={}, E={}, mse={}, h2z={})
    t0 = time.time()
    for name, hr in images(ds, a.limit):
        z = grey(hr); res['S'][name] = spec_S(z).tolist()
        for s in a.scales:
            lr = degrade(hr, s).to(dev); inp = (lr - 0.5) / 0.5; h = HR // s
            for full, abl in PAIRS:
                mf, ma = NET[full], NET[abl]
                mf.gen_feat(inp); ff = query(mf, 2.0 / HR).clamp(0, 1)
                fm = torch.stack([query(mf, 2.0 / (h * sp)).clamp(0, 1) for sp in SLIN]).mean(0)
                fl = torch.stack([query(mf, 2.0 / (h * sp)).clamp(0, 1) for sp in SLOG]).mean(0)
                ma.gen_feat(inp); a0 = query(ma, 2.0 / HR); a1 = query(ma, 2.0 / h); a4 = query(ma, 2.0 / (4 * h))
                fa = a0.clamp(0, 1)
                key = f'{full}|{s}'
                for tag, x in (('full', ff), ('abl', fa), ('marg', fm), ('mlog', fl)):
                    res['E'].setdefault(f'{key}|{tag}', {})[name] = spec_E(z, grey(x)).tolist()
                res['mse'].setdefault(key, {})[name] = dict(abl_marg=mse(fa, fm), abl_full=mse(fa, ff), abl_mlog=mse(fa, fl))
                res['h2z'].setdefault(key, {})[name] = max(float((a0 - a1).abs().max()), float((a0 - a4).abs().max()))
        print(ds, name, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(res, open(f'{OUT}/l2_{ds}.json', 'w'))
    print(ds, 'done', len(res['S']), 'images', flush=True)
