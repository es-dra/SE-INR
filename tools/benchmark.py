#!/usr/bin/env python3
"""Standard-benchmark evaluator for any checkpoint: PIL-bicubic LR, Y-PSNR with shave = scale (RGB, shave scale+6 on DIV2K).

  python3 tools/benchmark.py --ckpt save/liif/epoch-last.pth --name liif-s1 --data-root Data \
      --datasets set5 set14 bsd100 urban100 div2k --scales 2 3 4 6 8 12 --out bench.jsonl

Appends one JSON line per (name, dataset, scale). Same protocol as entrypoints/test.py; the feature map is cropped to the
input size for encoders that pad. Non-integer scales are not used here.
"""
import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = {'set5': 'Set5/HR', 'set14': 'Set14/HR', 'bsd100': 'BSD100/HR', 'urban100': 'Urban100/HR', 'div2k': 'DIV2K_valid_HR'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--data-root', default='Data')
    ap.add_argument('--datasets', nargs='+', default=['set5', 'set14', 'bsd100', 'urban100'])
    ap.add_argument('--scales', nargs='+', type=int, default=[2, 3, 4, 6, 8, 12])
    ap.add_argument('--bsize', type=int, default=30000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    import torch
    from torch.utils.data import DataLoader
    import datasets
    import models
    import utils

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model = models.make(ck['model'], load_sd=True).to(device).eval()
    norm_sub = torch.tensor([0.5], device=device).view(1, -1, 1, 1)
    done = set()
    if Path(args.out).exists():
        for line in open(args.out):
            r = json.loads(line)
            done.add((r['name'], r['dataset'], r['scale']))
    for ds in args.datasets:
        for s in args.scales:
            if (args.name, ds, s) in done:
                continue
            spec = {'name': 'sr-implicit-downsampled', 'args': {'scale_min': s, 'scale_max': s}}
            folder = datasets.make({'name': 'image-folder', 'args': {'root_path': str(Path(args.data_root) / DATASETS[ds])}})
            data = datasets.make(spec, args={'dataset': folder})
            loader = DataLoader(data, batch_size=1, num_workers=4, pin_memory=device == 'cuda')
            metric = partial(utils.calc_psnr, dataset='div2k' if ds == 'div2k' else 'benchmark', scale=s)
            avg = utils.Averager()
            t0 = time.time()
            with torch.no_grad():
                for batch in loader:
                    inp = (batch['inp'].to(device) - norm_sub) / 0.5
                    coord, cell = batch['coord'].to(device), batch['cell'].to(device)
                    model.gen_feat(inp)
                    model.feat = model.feat[:, :, :inp.shape[-2], :inp.shape[-1]]
                    preds = [model.query_rgb(coord[:, lo:lo + args.bsize], cell[:, lo:lo + args.bsize])
                             for lo in range(0, coord.shape[1], args.bsize)]
                    pred = (torch.cat(preds, dim=1) * 0.5 + 0.5).clamp_(0, 1)
                    ih, iw = inp.shape[-2:]
                    H, W = round(ih * s), round(iw * s)
                    pred = pred.view(1, H, W, 3).permute(0, 3, 1, 2)
                    gt = batch['gt'].to(device).view(1, H, W, 3).permute(0, 3, 1, 2)
                    avg.add(metric(pred, gt).item())
            rec = {'name': args.name, 'dataset': ds, 'scale': s, 'psnr': avg.item(), 'images': len(data),
                   'ckpt': args.ckpt, 'registry_name': ck['model']['name'], 'epoch': ck.get('epoch'),
                   'seconds': round(time.time() - t0, 1)}
            with open(args.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
