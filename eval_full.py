import os
import sys
import math
import json
import argparse
from functools import partial
from datetime import datetime

import yaml
import torch
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
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm, eval_type, eval_bsize, device):
    model.eval()
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

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


def make_test_config(benchmark, scale, data_root='/workspace/SE-INR/Data'):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output', type=str, default='eval_results.json')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    MODELS = {
        'LIIF': 'save/edsr-baseline-liif/epoch-best.pth',
        'LIIF-EQ': 'save/edsr-baseline-liif-EQ/epoch-best.pth',
        'SE-INR-unfold': 'save/se-inr_unfold/epoch-best.pth',
    }

    BENCHMARKS = ['Set5', 'Set14', 'BSD100', 'Urban100']
    ID_SCALES = [2, 3, 4]
    OOD_SCALES = [6, 8, 12, 16, 24, 30]

    results = {}
    if args.skip_existing and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            results = json.load(f)
        print(f"Loaded existing results from {args.output}")

    total = len(MODELS) * len(BENCHMARKS) * (len(ID_SCALES) + len(OOD_SCALES))
    done = 0

    for model_name, model_path in MODELS.items():
        if model_name not in results:
            results[model_name] = {}

        if not os.path.exists(model_path):
            print(f"[SKIP] {model_name}: {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_path})")
        print(f"{'='*60}")

        for benchmark in BENCHMARKS:
            if benchmark not in results[model_name]:
                results[model_name][benchmark] = {}

            all_scales = ID_SCALES + OOD_SCALES
            for scale in all_scales:
                scale_key = f'x{scale}'

                if args.skip_existing and scale_key in results[model_name][benchmark]:
                    print(f"  [SKIP] {benchmark} x{scale} (existing: {results[model_name][benchmark][scale_key]:.4f})")
                    done += 1
                    continue

                print(f"  Evaluating {benchmark} x{scale}...", end=' ', flush=True)
                t0 = datetime.now()
                try:
                    psnr = run_eval(model_path, benchmark, scale, device)
                    elapsed = (datetime.now() - t0).total_seconds()
                    results[model_name][benchmark][scale_key] = round(psnr, 4)
                    print(f"PSNR = {psnr:.4f} dB ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    results[model_name][benchmark][scale_key] = None

                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)

                done += 1

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")

    print_summary(results, ID_SCALES, OOD_SCALES, BENCHMARKS)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


def print_summary(results, id_scales, ood_scales, benchmarks):
    print(f"\n{'='*80}")
    print("ID Evaluation (In-Distribution: x2, x3, x4)")
    print(f"{'='*80}")
    header = f"{'Model':<15}"
    for s in id_scales:
        header += f"{'x'+str(s):>10}"
    print(header)
    print("-" * len(header))

    for model_name, model_results in results.items():
        row = f"{model_name:<15}"
        for s in id_scales:
            vals = []
            for b in benchmarks:
                v = model_results.get(b, {}).get(f'x{s}', None)
                if v is not None:
                    vals.append(v)
            if vals:
                avg = sum(vals) / len(vals)
                row += f"{avg:>10.4f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    print(f"\n{'='*80}")
    print("OOD Evaluation (Out-of-Distribution: x6, x8, x12, x16, x24, x30)")
    print(f"{'='*80}")
    header = f"{'Model':<15}"
    for s in ood_scales:
        header += f"{'x'+str(s):>10}"
    print(header)
    print("-" * len(header))

    for model_name, model_results in results.items():
        row = f"{model_name:<15}"
        for s in ood_scales:
            vals = []
            for b in benchmarks:
                v = model_results.get(b, {}).get(f'x{s}', None)
                if v is not None:
                    vals.append(v)
            if vals:
                avg = sum(vals) / len(vals)
                row += f"{avg:>10.4f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    print(f"\n{'='*80}")
    print("Per-Dataset Detail (All Scales)")
    print(f"{'='*80}")
    all_scales = id_scales + ood_scales
    for model_name, model_results in results.items():
        print(f"\n  {model_name}:")
        header = f"    {'Dataset':<12}"
        for s in all_scales:
            header += f"{'x'+str(s):>8}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for b in benchmarks:
            row = f"    {b:<12}"
            for s in all_scales:
                v = model_results.get(b, {}).get(f'x{s}', None)
                if v is not None:
                    row += f"{v:>8.2f}"
                else:
                    row += f"{'--':>8}"
            print(row)


if __name__ == '__main__':
    main()
