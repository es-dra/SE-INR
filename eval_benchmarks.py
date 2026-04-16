"""
Benchmark evaluation for three trained models:
  1. LIIF (baseline)     - save/edsr-baseline-liif/epoch-best.pth
  2. LIIF-EQ             - save/edsr-baseline-liif-EQ/epoch-best.pth
  3. SE-INR-S0           - save/se-inr-s0/epoch-best.pth

Evaluates on Set5, Set14, BSD100, Urban100 at scales x2, x3, x4.
"""
import os
import math
import json
from functools import partial

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
            preds.append(pred)
            ql = qr
        return torch.cat(preds, dim=1)


def eval_psnr(loader, model, data_norm, eval_type, eval_bsize=None, verbose=False):
    model.eval()
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    scale = int(eval_type.split('-')[1])
    metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)

    val_res = utils.Averager()
    for batch in tqdm(loader, leave=False, desc=f'val {eval_type}'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            pred = batched_predict(model, inp, coord, cell, eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        ih, iw = batch['inp'].shape[-2:]
        s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            print(f'  {val_res.item():.4f}')

    return val_res.item()


def make_loader(benchmark, scale, batch_size=1):
    root = f'/workspace/SE-INR/Data/{benchmark}/HR'
    spec = {
        'dataset': {
            'name': 'image-folder',
            'args': {'root_path': root, 'cache': 'none'}
        },
        'wrapper': {
            'name': 'sr-implicit-downsampled',
            'args': {
                'inp_size': None,
                'scale_min': scale,
                'scale_max': scale,
            }
        },
        'batch_size': batch_size
    }
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)


def eval_model(model_path, model_name, benchmarks, scales, data_norm):
    print(f'\n{"="*60}')
    print(f'Model: {model_name}')
    print(f'Checkpoint: {model_path}')
    print(f'{"="*60}')

    model_spec = torch.load(model_path, map_location='cuda')['model']
    model = models.make(model_spec, load_sd=True).cuda()
    print(f'Model loaded. Device: next(model.parameters()).device')

    results = {}
    for bench in benchmarks:
        results[bench] = {}
        for scale in scales:
            loader = make_loader(bench, scale)
            eval_type = f'benchmark-{scale}'
            psnr = eval_psnr(loader, model, data_norm, eval_type, eval_bsize=50000, verbose=False)
            results[bench][f'x{scale}'] = round(psnr, 4)
            print(f'  {bench:10s} x{scale}: PSNR = {psnr:.4f} dB')

    avg_results = {}
    for scale in scales:
        vals = [results[b][f'x{scale}'] for b in benchmarks]
        avg = sum(vals) / len(vals)
        avg_results[f'x{scale}'] = round(avg, 4)
        print(f'  {"AVG":10s} x{scale}: PSNR = {avg:.4f} dB')

    return results, avg_results


def main():
    base_dir = '/workspace/SE-INR/Equivariant-ASISR'
    os.chdir(base_dir)

    benchmarks = ['Set5', 'Set14', 'BSD100', 'Urban100']
    scales = [2, 3, 4]

    data_norm = {
        'inp': {'sub': [0.5], 'div': [0.5]},
        'gt': {'sub': [0.5], 'div': [0.5]}
    }

    models_to_eval = [
        ('save/edsr-baseline-liif/epoch-best.pth', 'LIIF (baseline)'),
        ('save/edsr-baseline-liif-EQ/epoch-best.pth', 'LIIF-EQ'),
        ('save/se-inr-s0/epoch-best.pth', 'SE-INR-S0'),
    ]

    all_results = {}
    for path, name in models_to_eval:
        full_path = os.path.join(base_dir, path)
        if not os.path.exists(full_path):
            print(f'WARNING: {full_path} not found, skipping')
            continue
        results, avg = eval_model(full_path, name, benchmarks, scales, data_norm)
        all_results[name] = {'by_benchmark': results, 'avg': avg}

    # Print summary table
    print(f'\n{"="*80}')
    print(f'BENCHMARK SUMMARY')
    print(f'{"="*80}')
    header = f'{"Model":<20s}' + ''.join([f'{"x"+str(s):>10s}' for s in scales]) + f'{"AVG":>10s}'
    print(header)
    print('-' * 80)
    for name, res in all_results.items():
        row = f'{name:<20s}'
        scale_avgs = []
        for scale in scales:
            vals = [res['by_benchmark'][b][f'x{scale}'] for b in benchmarks]
            avg = sum(vals) / len(vals)
            scale_avgs.append(avg)
            row += f'{avg:>10.4f}'
        overall = sum(scale_avgs) / len(scale_avgs)
        row += f'{overall:>10.4f}'
        print(row)

    # Save to JSON
    out_path = os.path.join(base_dir, 'eval_benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
