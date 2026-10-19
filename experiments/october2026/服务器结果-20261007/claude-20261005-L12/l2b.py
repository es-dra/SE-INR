# L2b re-render (prereg 预注册-L2b-20261005.md). L2 outputs kept only error spectra vs HR, not f_full/f_marg,
# so f_full and f_marg are re-rendered with exactly l2.py's settings (same images, scales, 31 linear cells, clamping).
# Saves per image per (pair, scale): Dk = per-k energy of demeaned grey(f_marg - f_full), Fk = per-k energy of demeaned grey(f_full).
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from common import HR, images, degrade, spec_S, grey, CK
ap = argparse.ArgumentParser()
ap.add_argument('--sets', nargs='+', default=['Set14', 'BSD100', 'Urban100'])
ap.add_argument('--scales', nargs='+', type=int, default=[2, 3, 4, 8])
ap.add_argument('--limit', type=int, default=30)
ap.add_argument('--outdir', required=True)
a = ap.parse_args()
OUT = os.path.abspath(a.outdir); os.makedirs(OUT, exist_ok=True)   # before chdir (l2.py bug)
ROOT = '/workspace/SE-INR/scale-equivariant-inr-pilot'; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/src'); os.chdir(ROOT)
import models, utils
dev = 'cuda'
FULL = ['lte', 'sc-inr']
SLIN = np.round(np.linspace(1.0, 4.0, 31), 6)
def load(mn): return models.make(torch.load(CK / 'seed1' / mn / 'epoch-last.pth', map_location='cpu', weights_only=False)['model'], load_sd=True).to(dev).eval()
NET = {mn: load(mn) for mn in FULL}
coord = utils.make_coord((HR, HR)).unsqueeze(0).to(dev)
@torch.no_grad()
def query(m, c):
    cell = torch.full_like(coord, float(c))
    return (m.query_rgb(coord, cell)[0] * 0.5 + 0.5).view(HR, HR, 3).permute(2, 0, 1)
for ds in a.sets:
    res = dict(meta=dict(HR=HR, scales=a.scales, slin=SLIN.tolist()), D={}, F={})
    t0 = time.time()
    for name, hr in images(ds, a.limit):
        for s in a.scales:
            lr = degrade(hr, s).to(dev); inp = (lr - 0.5) / 0.5; h = HR // s
            for full in FULL:
                m = NET[full]; m.gen_feat(inp)
                ff = query(m, 2.0 / HR).clamp(0, 1)
                fm = torch.stack([query(m, 2.0 / (h * sp)).clamp(0, 1) for sp in SLIN]).mean(0)
                key = f'{full}|{s}'
                res['D'].setdefault(key, {})[name] = spec_S(grey(fm) - grey(ff)).tolist()
                res['F'].setdefault(key, {})[name] = spec_S(grey(ff)).tolist()
        print(ds, name, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(res, open(f'{OUT}/l2b_{ds}.json', 'w'))
    print(ds, 'done', flush=True)
