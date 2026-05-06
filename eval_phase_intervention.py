"""
Causal intervention: zero out LTE's h_p(c) at inference time.
Tests whether h_p(c) is the direct cause of OOD degradation.

Compares:
  LTE (normal):  h_p(c) active
  LTE (phase=0): same checkpoint, h_p(c) forced to zero at inference
  LTE-NoCell:        separately trained without h_p(c)
"""

import os, sys, math, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import models, datasets, utils


def eval_psnr_for_model(model, loader, device, data_norm, scale):
    """Evaluate PSNR using standard benchmark metric (grayscale + shave)."""
    import math
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

    model.eval()
    val_res = utils.Averager()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc='eval'):
            for k, v in batch.items():
                batch[k] = v.to(device)
            inp = (batch['inp'] - inp_sub) / inp_div
            coord = batch['coord']
            cell = batch['cell']
            pred = model(inp, coord, cell)
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)

            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]

            res = utils.calc_psnr(pred, batch['gt'], dataset='benchmark', scale=scale)
            val_res.add(res.item(), inp.shape[0])
    return val_res.item()


def make_loader(benchmark, scale, data_root, batch_size=1):
    """Create a dataloader for benchmark at given scale."""
    import datasets
    from torch.utils.data import DataLoader

    spec = {
        'dataset': {'name': 'image-folder',
                    'args': {'root_path': f'{data_root}/{benchmark}/HR'}},
        'wrapper': {'name': 'sr-implicit-downsampled',
                    'args': {'scale_min': scale, 'scale_max': scale}},
        'batch_size': batch_size
    }
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


def main():
    device = 'cuda:0'
    data_root = '/workspace/SE-INR/Data'
    data_norm = {'inp': {'sub': [0.5], 'div': [0.5]},
                 'gt':  {'sub': [0.5], 'div': [0.5]}}

    # Load LTE
    lte_ckpt = torch.load('save/lte/epoch-best.pth',
                          map_location='cpu')
    lte_model = models.make(lte_ckpt['model'], load_sd=True).to(device)

    # Load LTE-NoCell
    noc_ckpt = torch.load('save/lte-no-cell/epoch-best.pth',
                          map_location='cpu')
    noc_model = models.make(noc_ckpt['model'], load_sd=True).to(device)

    benchmarks = ['Set5', 'Set14', 'BSD100', 'Urban100']
    id_scales = [2, 3, 4]
    ood_scales = [6, 8, 12, 16, 24, 30]

    all_results = {'LTE': {}, 'LTE-phase0': {}, 'LTE-NoCell': {}}

    for benchmark in benchmarks:
        print(f'\n{"="*55}')
        print(f'Benchmark: {benchmark}')
        print(f'{"="*55}')

        for scale in id_scales + ood_scales:
            loader = make_loader(benchmark, scale, data_root)

            # LTE (normal)
            lte_model.eval()
            psnr_lte = eval_psnr_for_model(lte_model, loader, device, data_norm, scale)

            # Reload loader (exhausted by first eval)
            loader = make_loader(benchmark, scale, data_root)

            # LTE (phase=0) — causal intervention: zero out phase weights
            orig_weight = lte_model.phase.weight.data.clone()
            lte_model.phase.weight.data.zero_()
            psnr_zero = eval_psnr_for_model(lte_model, loader, device, data_norm, scale)
            lte_model.phase.weight.data.copy_(orig_weight)  # Restore

            # Reload
            loader = make_loader(benchmark, scale, data_root)

            # LTE-NoCell
            psnr_noc = eval_psnr_for_model(noc_model, loader, device, data_norm, scale)

            tag = 'ID' if scale <= 4 else 'OOD'
            print(f'  x{scale:<3} {tag}:  LTE={psnr_lte:.2f}  '
                  f'LTE(φ=0)={psnr_zero:.2f}  NoC={psnr_noc:.2f}  '
                  f'Δ(φ=0-LTE)={psnr_zero-psnr_lte:+.2f}  '
                  f'Δ(NoC-LTE)={psnr_noc-psnr_lte:+.2f}')

            all_results['LTE'][f'{benchmark}_x{scale}'] = round(psnr_lte, 4)
            all_results['LTE-phase0'][f'{benchmark}_x{scale}'] = round(psnr_zero, 4)
            all_results['LTE-NoCell'][f'{benchmark}_x{scale}'] = round(psnr_noc, 4)

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY: Mean ΔPSNR vs LTE")
    print(f"{'='*65}")
    for condition, label in [('LTE-phase0', 'LTE(φ=0)-LTE'),
                              ('LTE-NoCell', 'LTE-NoCell-LTE')]:
        id_deltas, ood_deltas = [], []
        for b in benchmarks:
            for s in id_scales:
                k = f'{b}_x{s}'
                if k in all_results[condition] and k in all_results['LTE']:
                    id_deltas.append(all_results[condition][k] - all_results['LTE'][k])
            for s in ood_scales:
                k = f'{b}_x{s}'
                if k in all_results[condition] and k in all_results['LTE']:
                    ood_deltas.append(all_results[condition][k] - all_results['LTE'][k])
        print(f'  {label}:  ID mean={sum(id_deltas)/len(id_deltas):+.3f}  '
              f'OOD mean={sum(ood_deltas)/len(ood_deltas):+.3f}')

    # Save
    with open('results/phase_intervention.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nSaved to results/phase_intervention.json')


if __name__ == '__main__':
    main()
