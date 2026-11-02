#!/usr/bin/env python3
"""Stage 03: warping evaluation on trained checkpoints (inference only).

Three protocols share one geometry core (backward homography mapping with closed-form Jacobians):
  ltew-warp   LTEW/SRWarp format: warped LR inputs + forward 3x3 matrices (.pth) + HR ground truth; output grid = HR grid;
              masked PSNR over pixels whose source lies inside the input.
  ltew-asym   LTEW's 24 asymmetric scale pairs (or any list): LR by per-axis PIL bicubic as in LTEW; homography = scaling.
  minify      our minification split: input = HR, output = HR-sized grid under a homography with local minification;
              reference = 8x8 supersampled bicubic backward warp (box-filtered); isolates anti-aliasing at the output.

Each model is queried with a per-pixel footprint derived from the Jacobian J = d(LR row, col)/d(out row, col):
  token   LIIF/LTE/SC-INR as trained: axis-aligned cell = bounding box of the parallelogram J[-1/2,1/2]^2
  jac     SC-INR and SC-INR-exact: parallelogram footprint through the sinc response (query_rgb(..., jac=J))
Baselines: bicubic point warp and 8x8 supersampled bicubic (the reference's own construction, an upper bound).

  python3 warp.py --protocol minify --hr /data/Set14/HR --ckpt liif=/p/liif.pth lte=/p/lte.pth sc-inr=/p/sc.pth --out minify.jsonl
  python3 warp.py --protocol ltew-warp --hr load/benchmark/Set14/HR --warp-dir load/benchmark/Set14/LR_warp_in_scale --ckpt ... --out w.jsonl
  python3 warp.py --protocol ltew-asym --hr /data/B100/HR --pairs 4.00,1.40 1.50,3.00 --ckpt ... --out asym.jsonl
  python3 warp.py --smoke /tmp/warp-smoke
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / 'src' if (HERE / 'src').exists() else HERE.parents[2] / 'src'
sys.path.insert(0, str(SRC))

Y_COEFFS = (65.738 / 256, 129.057 / 256, 25.064 / 256)
LTEW_PAIRS = {  # (vertical, horizontal) as encoded in LTEW's test configs; first three in-scale, last three out-of-scale
    'set5': [(1.5, 4.0), (1.5, 3.5), (1.6, 3.05), (3.0, 8.0), (3.0, 7.0), (3.2, 6.1)],
    'set14': [(4.0, 2.0), (3.5, 2.0), (3.5, 1.75), (8.0, 4.0), (7.0, 4.0), (7.0, 3.5)],
    'bsd100': [(4.0, 1.4), (1.5, 3.0), (3.5, 1.45), (8.0, 2.8), (3.0, 6.0), (7.0, 2.9)],
    'urban100': [(1.6, 3.0), (1.6, 3.8), (3.55, 1.55), (3.2, 6.0), (3.2, 7.6), (7.1, 3.1)],
}


# ----------------------------------------------------------------------------- geometry
def backward_map(m, H, W, h, w, device):
    """Output pixel centres (row r, col c) -> source coordinates under the forward matrix m (input px -> output px, cv2 order x=col, y=row,
    integer pixel centres). Returns source (y, x) in input pixel-index units [H, W, 2], Jacobian d(y_src, x_src)/d(r, c) [H, W, 2, 2], mask."""
    Minv = torch.linalg.inv(torch.as_tensor(m, dtype=torch.float64)).to(device)
    r = torch.arange(H, dtype=torch.float64, device=device)
    c = torch.arange(W, dtype=torch.float64, device=device)
    cc, rr = torch.meshgrid(c, r, indexing='xy')          # [H, W]
    ones = torch.ones_like(cc)
    P = torch.stack([cc, rr, ones], -1) @ Minv.T           # [H, W, 3]: (a, b, d)
    a, b, d = P[..., 0], P[..., 1], P[..., 2]
    x, y = a / d, b / d
    # derivatives: d(a/d)/dc = (M00 d - a M20)/d^2, d(a/d)/dr = (M01 d - a M21)/d^2 ; same for b
    dxdc = (Minv[0, 0] * d - a * Minv[2, 0]) / d ** 2
    dxdr = (Minv[0, 1] * d - a * Minv[2, 1]) / d ** 2
    dydc = (Minv[1, 0] * d - b * Minv[2, 0]) / d ** 2
    dydr = (Minv[1, 1] * d - b * Minv[2, 1]) / d ** 2
    J = torch.stack([torch.stack([dydr, dydc], -1), torch.stack([dxdr, dxdc], -1)], -2)   # [H, W, 2, 2] rows (y, x), cols (r, c)
    mask = (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5) & (d > 0)
    src = torch.stack([y, x], -1)
    return src.float(), J.float(), mask


def scaling_matrix(sy, sx):
    """Forward matrix for pure per-axis scaling with integer pixel centres: x_out = sx*(x_in + 0.5) - 0.5."""
    return np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])


def random_minifying_homography(h, w, rng, strength):
    """Forward matrix (input px -> output px) of a perspective tilt that compresses one side of the image.
    strength in (0, 1): max local minification ratio grows with it; the output grid has the input's size."""
    # perspective term along x, plus a compensating scale so the image stays inside the output frame
    px = strength / w
    # map the input corners and fit a mild similarity so that the warped content covers the frame
    m = np.array([[1.0, 0, 0], [0, 1.0, 0], [px, 0, 1.0]])
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], dtype=float) @ m.T
    corners = corners[:, :2] / corners[:, 2:3]
    sx = (w - 1) / (corners[:, 0].max() - corners[:, 0].min())
    sy = (h - 1) / (corners[:, 1].max() - corners[:, 1].min())
    fit = np.array([[sx, 0, -sx * corners[:, 0].min()], [0, sy, -sy * corners[:, 1].min()], [0, 0, 1]])
    rot = rng.uniform(-0.15, 0.15)
    R = np.array([[math.cos(rot), -math.sin(rot), (w - 1) / 2], [math.sin(rot), math.cos(rot), (h - 1) / 2], [0, 0, 1]]) @ \
        np.array([[1, 0, -(w - 1) / 2], [0, 1, -(h - 1) / 2], [0, 0, 1]])
    return R @ fit @ m


# ----------------------------------------------------------------------------- sampling
def grid_sample_src(img, src, mode='bicubic'):
    """img [1,3,h,w], src [N,2] (y, x) pixel-index coords -> [N,3]."""
    import torch.nn.functional as F
    h, w = img.shape[-2:]
    gx = (src[:, 1] + 0.5) / w * 2 - 1
    gy = (src[:, 0] + 0.5) / h * 2 - 1
    g = torch.stack([gx, gy], -1).view(1, -1, 1, 2)
    return F.grid_sample(img, g, mode=mode, padding_mode='border', align_corners=False)[0, :, :, 0].T


def supersampled_reference(img, m, H, W, n, device):
    """Box-filtered reference: average of n x n bicubic point samples per output pixel (sub-pixel offsets within the pixel)."""
    acc = torch.zeros(H * W, 3, device=device)
    offs = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n - 0.5
    Minv = torch.linalg.inv(torch.as_tensor(m, dtype=torch.float64)).to(device)
    r = torch.arange(H, dtype=torch.float64, device=device)
    c = torch.arange(W, dtype=torch.float64, device=device)
    cc, rr = torch.meshgrid(c, r, indexing='xy')
    for oy in offs:
        for ox in offs:
            P = torch.stack([cc + ox, rr + oy, torch.ones_like(cc)], -1) @ Minv.T
            src = torch.stack([P[..., 1] / P[..., 2], P[..., 0] / P[..., 2]], -1).reshape(-1, 2).float()
            acc += grid_sample_src(img, src)
    return acc / (n * n)


# ----------------------------------------------------------------------------- models
def load_model(path, device, override=()):
    import models
    ck = torch.load(path, map_location='cpu', weights_only=True)
    for item in override:
        k, v = item.split('=', 1)
        ck['model']['args'][k] = {'true': True, 'false': False}.get(v.lower(), v)
    return models.make(ck['model'], load_sd=True).to(device).eval(), ck['model']['name']


def predict(model, inp, src, J, mode, chunk=20000):
    """Query a LIIF-interface model at source coords with footprints from J. mode 'token' (axis-aligned bounding-box cell) or 'jac'."""
    h, w = inp.shape[-2:]
    coord = torch.stack([(src[:, 0] + 0.5) / h * 2 - 1, (src[:, 1] + 0.5) / w * 2 - 1], -1)
    ext_y = J[:, 0, 0].abs() + J[:, 0, 1].abs()           # LR-row extent of the parallelogram
    ext_x = J[:, 1, 0].abs() + J[:, 1, 1].abs()
    cell = torch.stack([2 * ext_y / h, 2 * ext_x / w], -1)  # LIIF convention: rel_cell = cell * feat.shape = 2 * extent
    out = []
    with torch.no_grad():
        model.gen_feat(inp)
        for lo in range(0, coord.shape[0], chunk):
            q, cl = coord[lo:lo + chunk][None], cell[lo:lo + chunk][None]
            if mode == 'jac':
                out.append(model.query_rgb(q, cl, jac=J[lo:lo + chunk][None])[0])
            else:
                out.append(model.query_rgb(q, cl)[0])
    return torch.cat(out, 0) * 0.5 + 0.5


# ----------------------------------------------------------------------------- metrics
def psnrs(pred, gt, mask):
    """Masked PSNR on RGB and on Y (BT.601), pred/gt [N,3] in [0,1], mask [N] bool."""
    pred, gt = pred.clamp(0, 1).double(), gt.double()
    mk = mask.double()
    n = mk.sum().clamp_min(1)
    rgb = ((pred - gt) ** 2 * mk[:, None]).sum() / (3 * n)
    wy = torch.tensor(Y_COEFFS, dtype=torch.float64, device=pred.device)
    ylum = (((pred @ wy) - (gt @ wy)) ** 2 * mk).sum() / n
    return {'psnr_rgb': float(-10 * torch.log10(rgb)), 'psnr_y': float(-10 * torch.log10(ylum)), 'valid_fraction': float(mk.mean())}


# ----------------------------------------------------------------------------- protocols
def load_png(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255


def run_case(models_, hr, inp, m, H, W, device, record, out, gt_override=None, ss=0):
    """One image under one forward matrix m: input `inp` [1,3,h,w] (0..1), ground truth = hr (or override) on the H x W output grid."""
    h, w = inp.shape[-2:]
    src, J, mask = backward_map(m, H, W, h, w, device)
    src, J, mask = src.reshape(-1, 2), J.reshape(-1, 2, 2), mask.reshape(-1)
    gt = gt_override if gt_override is not None else hr.reshape(-1, 3)
    det = (J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]).abs()
    geom = {'valid_fraction': float(mask.float().mean()), 'local_scale_median': float((1 / det.clamp_min(1e-9)).sqrt()[mask].median()),
            'minified_fraction': float((det[mask] > 1).float().mean())}
    rows = []
    bic = grid_sample_src(inp, src)
    rows.append({'model': 'bicubic-point', 'mode': 'baseline', **psnrs(bic, gt, mask)})
    if ss:
        rows.append({'model': f'bicubic-ss{ss}', 'mode': 'baseline', **psnrs(supersampled_reference(inp, m, H, W, ss, device), gt, mask)})
    x = inp * 2 - 1
    for name, (model, reg) in models_.items():
        modes = ['token'] + (['jac'] if reg.startswith('sc_inr') else [])
        for mode in modes:
            t0 = time.time()
            pred = predict(model, x, src, J, mode)
            rows.append({'model': name, 'registry': reg, 'mode': mode, **psnrs(pred, gt, mask), 'seconds': round(time.time() - t0, 2)})
    with open(out, 'a') as f:
        for r in rows:
            f.write(json.dumps({**record, **geom, **r}) + '\n')
    return rows


def protocol_ltew_warp(args, models_, device):
    hr_files = sorted(Path(args.hr).glob('*.png'))
    warp = Path(args.warp_dir)
    pngs = sorted(p for p in warp.iterdir() if p.suffix == '.png')
    mats = sorted(p for p in warp.iterdir() if p.suffix == '.pth')
    assert len(pngs) == len(mats) == len(hr_files), (len(pngs), len(mats), len(hr_files))
    for hp, lp, mp in zip(hr_files, pngs, mats):
        hr = torch.from_numpy(load_png(hp)).to(device)
        inp = torch.from_numpy(load_png(lp)).permute(2, 0, 1)[None].to(device)
        m = torch.load(mp, map_location='cpu', weights_only=False)
        m = np.asarray(m.numpy() if hasattr(m, 'numpy') else m, dtype=np.float64)
        if args.invert_matrix:
            m = np.linalg.inv(m)
        H, W = hr.shape[:2]
        run_case(models_, hr, inp, m, H, W, device, {'protocol': 'ltew-warp', 'split': warp.name, 'image': hp.stem}, args.out)
        print(hp.stem, flush=True)


def protocol_ltew_asym(args, models_, device):
    from torchvision import transforms
    pairs = [tuple(float(v) for v in p.split(',')) for p in args.pairs]
    for hp in sorted(Path(args.hr).glob('*.png')):
        img = Image.open(hp).convert('RGB')
        for sy, sx in pairs:
            W0, H0 = img.size
            h, w = math.floor(H0 / sy + 1e-9), math.floor(W0 / sx + 1e-9)
            H, W = round(h * sy), round(w * sx)
            hr_img = img.crop((0, 0, W, H))
            lr = transforms.ToTensor()(transforms.Resize((h, w), Image.BICUBIC)(hr_img))[None].to(device)
            hr = torch.from_numpy(np.asarray(hr_img, dtype=np.float32) / 255).to(device)
            m = scaling_matrix(H / h, W / w)
            run_case(models_, hr, lr, m, H, W, device, {'protocol': 'ltew-asym', 'sy': sy, 'sx': sx, 'image': hp.stem,
                                                       'in_scale': max(sy, sx) <= 4}, args.out)
        print(hp.stem, flush=True)


def protocol_minify(args, models_, device):
    rng = np.random.default_rng(args.seed)
    for hp in sorted(Path(args.hr).glob('*.png')):
        arr = load_png(hp)
        H, W = arr.shape[:2]
        if args.crop:
            t, l = (H - args.crop) // 2, (W - args.crop) // 2
            arr = arr[t:t + args.crop, l:l + args.crop]
            H, W = arr.shape[:2]
        inp = torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)
        for strength in args.strengths:
            m = random_minifying_homography(H, W, rng, strength)
            gt = supersampled_reference(inp, m, H, W, args.ss, device)
            run_case(models_, None, inp, m, H, W, device, {'protocol': 'minify', 'strength': strength, 'image': hp.stem, 'ss_ref': args.ss},
                     args.out, gt_override=gt, ss=0)
        print(hp.stem, flush=True)


def summarize(path):
    import collections
    acc = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        key = (r['protocol'], r.get('split') or r.get('strength') or ('in' if r.get('in_scale') else 'out'), r['model'], r['mode'])
        acc[key].append(r['psnr_y'])
    print(f"{'protocol':10s} {'split':10s} {'model':16s} {'mode':8s} {'n':>4s} {'PSNR-Y':>8s}")
    for k in sorted(acc, key=str):
        print(f'{k[0]:10s} {str(k[1]):10s} {k[2]:16s} {k[3]:8s} {len(acc[k]):4d} {np.mean(acc[k]):8.3f}')


# ----------------------------------------------------------------------------- smoke
def smoke(root):
    import models
    root = Path(root)
    (root / 'hr').mkdir(parents=True)
    (root / 'warp').mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        yy, xx = np.mgrid[:72, :96]
        img = (0.5 + 0.3 * (np.sin(yy / 4 + i) * np.cos(xx / 6))[..., None] + 0.1 * rng.standard_normal((72, 96, 3))).clip(0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(root / 'hr' / f'{i}.png')
        m = scaling_matrix(2.0, 3.0) @ random_minifying_homography(36, 32, rng, 0.3)   # input 36x32 -> output 72x96
        Image.fromarray((img[::2, ::3] * 255).astype(np.uint8)).save(root / 'warp' / f'{i}.png')
        torch.save(torch.tensor(m), root / 'warp' / f'{i}.pth')
    enc = {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}
    specs = {'liif': {'name': 'liif', 'args': {'encoder_spec': enc, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256] * 4}}}},
             'sc-inr': {'name': 'sc_inr_signed_phiz', 'args': {'encoder_spec': enc, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256] * 3}},
                        'hidden_dim': 256, 'num_freqs': 128, 'num_angles': 8, 'freq_min': 0.1, 'freq_max': 2.0, 'omega_bound': 2.1, 'learn_phase': True}},
             'exact': {'name': 'sc_inr_exact', 'args': {'encoder_spec': enc, 'coef_hidden': [128]}}}
    cks = []
    for k, spec in specs.items():
        mdl = models.make(spec)
        torch.save({'model': dict(spec, sd=mdl.state_dict()), 'epoch': 1}, root / f'{k}.pth')
        cks.append(f'{k}={root}/{k}.pth')
    base = [sys.executable, str(HERE / 'warp.py'), '--device', 'cpu', '--ckpt', *cks]
    subprocess.check_call(base + ['--protocol', 'ltew-warp', '--hr', str(root / 'hr'), '--warp-dir', str(root / 'warp'), '--out', str(root / 'w.jsonl')])
    subprocess.check_call(base + ['--protocol', 'ltew-asym', '--hr', str(root / 'hr'), '--pairs', '1.5,3.0', '--out', str(root / 'a.jsonl')])
    subprocess.check_call(base + ['--protocol', 'minify', '--hr', str(root / 'hr'), '--strengths', '0.3', '0.6', '--ss', '4', '--out', str(root / 'm.jsonl')])
    for f in ('w', 'a', 'm'):
        summarize(root / f'{f}.jsonl')
    print('smoke ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['ltew-warp', 'ltew-asym', 'minify'])
    ap.add_argument('--hr')
    ap.add_argument('--warp-dir')
    ap.add_argument('--pairs', nargs='*', default=[])
    ap.add_argument('--strengths', nargs='*', type=float, default=[0.3, 0.6, 0.9])
    ap.add_argument('--ss', type=int, default=8)
    ap.add_argument('--crop', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--ckpt', nargs='*', default=[], help='name=path')
    ap.add_argument('--override', nargs='*', default=[])
    ap.add_argument('--invert-matrix', action='store_true', help='if the stored matrices map output px -> input px')
    ap.add_argument('--out')
    ap.add_argument('--device', default=None)
    ap.add_argument('--summarize')
    ap.add_argument('--smoke')
    a = ap.parse_args()
    if a.smoke:
        smoke(a.smoke)
        sys.exit(0)
    if a.summarize:
        summarize(a.summarize)
        sys.exit(0)
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    models_ = {}
    for item in a.ckpt:
        name, path = item.split('=', 1)
        models_[name] = load_model(path, device, a.override)
    {'ltew-warp': protocol_ltew_warp, 'ltew-asym': protocol_ltew_asym, 'minify': protocol_minify}[a.protocol](a, models_, device)
