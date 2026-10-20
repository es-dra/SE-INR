"""Frequency probe for arbitrary-scale SR models.

For each image and scale it reconstructs the HR image, splits the error
spectrum into bins of the LR-normalised frequency u = 2*s*r (Chebyshev
radius r in cycles per HR pixel), and optionally sweeps the cell size.

Outputs one JSON row per (image, scale, u-bin):
    err[m]  squared error in the bin with the cell multiplied by m
    tgt     target energy in the bin
from which band recovery R(u) = 1 - err[1]/tgt and the cost of a
mis-set cell, err[m] - err[1], follow.

Example:
    python tools/frequency_probe.py --ckpt save/liif/epoch-last.pth \
        --data Data/DIV2K_valid_HR --crop 240 --scales 2 3 4 6 \
        --cell-mult 1 2 0.5 --out probe.json
    python tools/frequency_probe.py --summary probe.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import models  # noqa: E402
from datasets.gauss_wrapper import gauss_down  # noqa: E402

BIN = 0.25


def degrade(x, size, kind):
    """x: 1x3xHxW in [0,1]."""
    if kind == 'bicubic':
        return F.interpolate(x, size=size, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    if kind == 'gauss':
        return gauss_down(x[0], size)[None]
    raise ValueError(kind)


def load_image(path, crop, mult):
    im = np.asarray(Image.open(path).convert('RGB'), dtype='float32') / 255
    h, w = im.shape[:2]
    if crop:
        assert min(h, w) >= crop, path
        t, l = (h - crop) // 2, (w - crop) // 2
        return im[t:t + crop, l:l + crop]
    return im[:(h // mult) * mult, :(w // mult) * mult]


def u_bins(h, w, s):
    fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
    r = np.maximum(abs(fy)[:, None], abs(fx)[None, :])
    u = 2 * s * r
    return [(float(lo), (u >= lo) & (u < lo + BIN) & (r > 0)) for lo in np.arange(0, s, BIN)]


@torch.no_grad()
def predict(model, lr, hh, ww, cell_mult, needs_scale, scale, chunk, dev):
    vy, vx = (np.arange(hh) + .5) / hh, (np.arange(ww) + .5) / ww
    q = torch.from_numpy((2 * np.stack(np.meshgrid(vy, vx, indexing='ij'), -1).reshape(-1, 2) - 1).astype('float32'))[None].to(dev)
    model.gen_feat((lr * 2 - 1).to(dev))
    if hasattr(model, 'feat'):  # encoders that pad (e.g. SwinIR) must be cropped back to the LR grid
        model.feat = model.feat[:, :, :lr.shape[-2], :lr.shape[-1]]
    out = {}
    for m in cell_mult:
        cell = torch.ones_like(q)
        cell[..., 0] *= 2 * m / hh
        cell[..., 1] *= 2 * m / ww
        if needs_scale:
            z = model.query_rgb(q, torch.full((1,), float(scale), device=dev), cell)[0]
        else:
            z = torch.cat([model.query_rgb(q[:, i:i + chunk], cell[:, i:i + chunk])[0] for i in range(0, q.shape[1], chunk)])
        out[m] = (z.float().cpu().numpy() * .5 + .5).reshape(hh, ww, 3)
    return out


def run(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model = models.make(ck['model'], load_sd=True).to(dev).eval()
    lcm = int(np.lcm.reduce(a.scales))
    rows = []
    for path in sorted(glob.glob(os.path.join(a.data, '*.png')))[:a.limit or None]:
        x = load_image(path, a.crop, lcm)
        hh, ww = x.shape[:2]
        sh = a.shave
        T = np.fft.fft2(x[sh:hh - sh, sh:ww - sh].astype('float64'), axes=(0, 1))
        nrm = (T.shape[0] * T.shape[1]) ** 2 * 3
        xt = torch.from_numpy(x.transpose(2, 0, 1))[None]
        for s in a.scales:
            lr = degrade(xt, (hh // s, ww // s), a.degradation)
            preds = predict(model, lr, hh, ww, a.cell_mult, a.needs_scale, s, a.chunk, dev)
            S = {m: np.fft.fft2(p[sh:hh - sh, sh:ww - sh].astype('float64'), axes=(0, 1)) for m, p in preds.items()}
            for lo, mk in u_bins(hh - 2 * sh, ww - 2 * sh, s):
                rows.append(dict(img=os.path.basename(path)[:-4], s=s, lo=lo,
                                 tgt=float((abs(T[mk]) ** 2).sum() / nrm),
                                 err={str(m): float((abs(T[mk] - S[m][mk]) ** 2).sum() / nrm) for m in S}))
        print(os.path.basename(path), flush=True)
    json.dump(dict(ckpt=a.ckpt, data=a.data, crop=a.crop, degradation=a.degradation, rows=rows), open(a.out, 'w'))


def summary(path):
    d = json.load(open(path))
    err, tgt = defaultdict(Counter), Counter()
    for r in d['rows']:
        k = (r['s'], r['lo'])
        err[k].update(r['err'])
        tgt[k] += r['tgt']
    for s in sorted({k[0] for k in tgt}):
        ks = sorted(k for k in tgt if k[0] == s)
        print(f'x{s}  R(u): ' + ' '.join(f'{k[1]:.2f}:{1 - err[k]["1.0"] / tgt[k]:.3f}' for k in ks))
        nat = sum(err[k]['1.0'] for k in ks)
        for m in sorted(err[ks[0]], key=float):
            if m != '1.0':
                print(f'      cell x{m}: {10 * np.log10(sum(err[k][m] for k in ks) / nat):+.3f} dB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt')
    ap.add_argument('--data')
    ap.add_argument('--out', default='probe.json')
    ap.add_argument('--crop', type=int, default=240, help='centre crop size; 0 = full image')
    ap.add_argument('--scales', type=int, nargs='+', default=[2, 3, 4, 6])
    ap.add_argument('--cell-mult', type=float, nargs='+', default=[1.0])
    ap.add_argument('--degradation', choices=['bicubic', 'gauss'], default='bicubic')
    ap.add_argument('--shave', type=int, default=4)
    ap.add_argument('--needs-scale', action='store_true', help='model.query_rgb takes a scale argument (GaussianSR)')
    ap.add_argument('--chunk', type=int, default=60000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--summary', metavar='JSON', help='print R(u) and cell costs from an output file')
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    else:
        if 1.0 not in a.cell_mult:
            a.cell_mult = [1.0] + a.cell_mult
        run(a)
