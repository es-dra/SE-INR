#!/usr/bin/env python3
"""
Clean benchmark evaluation: LIIF vs SE-INR-unfold
Only 2 models, ID (x2,x3,x4) and OOD (x6,x8,x12,x16,x24,x30) scales.
Uses test.py style (config-based dataset) with eval_type='benchmark-N'.

FIX: metric_fn now uses dataset='benchmark', scale=N (shave + grayscale)
to match benchmark evaluation standard.
"""
import os, json, sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import math

import datasets
import models
import utils


def make_test_config(benchmark, scale, data_root='/workspace/SE-INR/Data'):
    """Build test config same as test.py uses."""
    return {
        'test_dataset': {
            'dataset': {
                'name': 'image-folder',
                'args': {
                    'root_path': f'{data_root}/{benchmark}/HR'
                }
            },
            'wrapper': {
                'name': 'sr-implicit-downsampled',
                'args': {
                    'scale_min': scale,
                    'scale_max': scale
                }
            },
            'batch_size': 1
        },
        'data_norm': {
            'inp': {'sub': [0.5], 'div': [0.5]},
            'gt': {'sub': [0.5], 'div': [0.5]}
        },
        'eval_type': f'benchmark-{scale}',
        'eval_bsize': 50000
    }


def batched_predict(model, inp, coord, cell, bsize):
    """Same as test.py / eval_full.py."""
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


def eval_psnr(loader, model, data_norm, eval_type, eval_bsize, device):
    """Compute PSNR on a loader. Uses same logic as test.py."""
    model.eval()

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

    def metric_fn(pred, gt):
        return utils.calc_psnr(pred, gt, dataset='benchmark', scale=int(eval_type.split('-')[1]))

    val_res = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.to(device)

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

        if eval_type is not None:
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])
        pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


def run_eval(model_path, benchmark, scale, device, data_root='/workspace/SE-INR/Data'):
    config = make_test_config(benchmark, scale, data_root)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=4, pin_memory=True)

    model_spec = torch.load(model_path, map_location='cpu')['model']
    model = models.make(model_spec, load_sd=True, strict=False).to(device)

    psnr = eval_psnr(loader, model,
                     data_norm=config.get('data_norm'),
                     eval_type=config.get('eval_type'),
                     eval_bsize=config.get('eval_bsize'),
                     device=device)

    del model
    torch.cuda.empty_cache()
    return psnr


def main():
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    # Two models only
    MODELS = {
        'LIIF':   'save/edsr-baseline-liif/epoch-best.pth',
        'SE-INR-unfold': 'save/se-inr_unfold/epoch-best.pth',
    }

    BENCHMARKS = ['Set5', 'Set14', 'BSD100', 'Urban100']
    ID_SCALES  = [2, 3, 4]
    OOD_SCALES = [6, 8, 12, 16, 24, 30]

    out_file = 'eval_clean_results_v4.json'

    results = {}
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded existing results from {out_file}")

    for model_name, model_path in MODELS.items():
        if model_name not in results:
            results[model_name] = {}

        if not os.path.exists(model_path):
            print(f"[SKIP] {model_name}: {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_path})")
        print(f"Loaded at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        for benchmark in BENCHMARKS:
            if benchmark not in results[model_name]:
                results[model_name][benchmark] = {}

            all_scales = ID_SCALES + OOD_SCALES
            for scale in all_scales:
                scale_key = f'x{scale}'
                existing = results[model_name][benchmark].get(scale_key)
                if existing is not None:
                    print(f"  [SKIP] {benchmark} x{scale} (existing: {existing:.4f})")
                    continue

                print(f"  {benchmark} x{scale}...", end=' ', flush=True)
                t0 = datetime.now()
                try:
                    psnr = run_eval(model_path, benchmark, scale, device)
                    elapsed = (datetime.now() - t0).total_seconds()
                    results[model_name][benchmark][scale_key] = round(psnr, 4)
                    print(f"PSNR = {psnr:.4f} dB ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results[model_name][benchmark][scale_key] = None

                with open(out_file, 'w') as f:
                    json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")

    print_summary(results, ID_SCALES, OOD_SCALES, BENCHMARKS)

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


def print_summary(results, id_scales, ood_scales, benchmarks):
    # Average across benchmarks
    print(f"\n{'='*80}")
    print(f"{'Average PSNR (4 benchmarks)':^80}")
    print(f"{'='*80}")

    all_id = id_scales + ood_scales

    # Header
    header = f"{'Model':<16}" + "".join([f"{'x'+str(s):>9}" for s in all_id])
    sep = "-" * (16 + 9 * len(all_id))
    print(header)
    print(sep)

    for model_name, model_results in results.items():
        row = f"{model_name:<16}"
        for s in all_id:
            vals = []
            for b in benchmarks:
                v = model_results.get(b, {}).get(f'x{s}', None)
                if v is not None:
                    vals.append(v)
            if vals:
                avg = sum(vals) / len(vals)
                row += f"{avg:>9.2f}"
            else:
                row += f"{'--':>9}"
        print(row)

    print(f"\n{'='*80}")
    print(f"{'Per-Dataset Detail':^80}")
    print(f"{'='*80}")

    for benchmark in benchmarks:
        print(f"\n  {benchmark}:")
        header = f"    {'Model':<16}" + "".join([f"{'x'+str(s):>8}" for s in all_id])
        print(header)
        print("    " + "-" * (16 + 8 * len(all_id)))
        for model_name, model_results in results.items():
            row = f"    {model_name:<16}"
            for s in all_id:
                v = model_results.get(benchmark, {}).get(f'x{s}', None)
                if v is not None:
                    row += f"{v:>8.2f}"
                else:
                    row += f"{'--':>8}"
            print(row)

    # Delta table
    print(f"\n{'='*80}")
    print(f"{'SE-INR-unfold vs LIIF Delta (dB)':^80}")
    print(f"{'='*80}")
    header = f"{'Scale':<12}" + "".join([f"{b:>10}" for b in benchmarks]) + f"{'Avg':>10}"
    print(header)
    print(" " + "-" * (12 + 10 * len(benchmarks) + 10))
    for s in all_id:
        scale_key = f'x{s}'
        row = f"{scale_key:<12}"
        deltas = []
        for b in benchmarks:
            v_se = results.get('SE-INR-unfold', {}).get(b, {}).get(scale_key, None)
            v_liif = results.get('LIIF', {}).get(b, {}).get(scale_key, None)
            if v_se is not None and v_liif is not None:
                d = v_se - v_liif
                row += f"{d:>+10.2f}"
                deltas.append(d)
            else:
                row += f"{'--':>10}"
        if deltas:
            avg_d = sum(deltas) / len(deltas)
            row += f"{avg_d:>+10.2f}"
        else:
            row += f"{'--':>10}"
        print(row)


if __name__ == '__main__':
    main()