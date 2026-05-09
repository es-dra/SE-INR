"""
Causal intervention: zero out LTE's h_p(c) at inference time.
Tests whether h_p(c) is the direct cause of OOD degradation.

Compares:
  LTE (normal):  h_p(c) active
  LTE (phase=0): same checkpoint, h_p(c) forced to zero at inference
  LTE-NoCellPhase:   separately trained without h_p(c)
"""

import os, sys, math, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def parse_csv(value, cast=str):
    return [cast(v.strip()) for v in value.split(',') if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the LTE phase(cell) intervention diagnostic."
    )
    parser.add_argument('--device', default='0',
                        help='CUDA device id, or "cpu". Default: 0')
    parser.add_argument('--save_root', default='save',
                        help='Checkpoint root containing lte/ and lte-nocellphase/.')
    parser.add_argument('--data_root', default=None,
                        help='Benchmark data root. Default: $SEINR_DATA_ROOT or ../Data.')
    parser.add_argument('--output', default='results/phase_intervention.json',
                        help='Output JSON path.')
    parser.add_argument('--benchmarks', default='Set5,Set14,BSD100,Urban100',
                        help='Comma-separated benchmark names.')
    parser.add_argument('--scales', default='2,3,4,6,8,12,16,24,30',
                        help='Comma-separated scale factors.')
    args = parser.parse_args()

    if args.device == 'cpu' or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = f'cuda:{args.device}'

    data_root = args.data_root
    if data_root is None:
        data_root = str(Path(os.environ.get('SEINR_DATA_ROOT', ROOT.parent / 'Data')))
    data_norm = {'inp': {'sub': [0.5], 'div': [0.5]},
                 'gt':  {'sub': [0.5], 'div': [0.5]}}

    save_root = Path(args.save_root)

    # Load LTE
    lte_ckpt = torch.load(save_root / 'lte' / 'epoch-best.pth',
                          map_location='cpu')
    lte_model = models.make(lte_ckpt['model'], load_sd=True).to(device)

    # Load LTE-NoCellPhase
    noc_ckpt = torch.load(save_root / 'lte-nocellphase' / 'epoch-best.pth',
                          map_location='cpu')
    noc_model = models.make(noc_ckpt['model'], load_sd=True).to(device)

    benchmarks = parse_csv(args.benchmarks)
    scales = parse_csv(args.scales, int)
    id_scales = [s for s in scales if s <= 4]
    ood_scales = [s for s in scales if s > 4]

    all_results = {'LTE': {}, 'LTE-phase0': {}, 'LTE-NoCellPhase': {}}

    for benchmark in benchmarks:
        print(f'\n{"="*55}')
        print(f'Benchmark: {benchmark}')
        print(f'{"="*55}')

        for scale in scales:
            loader = make_loader(benchmark, scale, data_root)

            # LTE (normal)
            lte_model.eval()
            psnr_lte = eval_psnr_for_model(lte_model, loader, device, data_norm, scale)

            # Reload loader (exhausted by first eval)
            loader = make_loader(benchmark, scale, data_root)

            # LTE (phase=0) — causal intervention: zero out phase weights
            orig_weight = lte_model.phase.weight.data.clone()
            orig_bias = None
            if lte_model.phase.bias is not None:
                orig_bias = lte_model.phase.bias.data.clone()
            lte_model.phase.weight.data.zero_()
            if lte_model.phase.bias is not None:
                lte_model.phase.bias.data.zero_()
            psnr_zero = eval_psnr_for_model(lte_model, loader, device, data_norm, scale)
            lte_model.phase.weight.data.copy_(orig_weight)  # Restore
            if orig_bias is not None:
                lte_model.phase.bias.data.copy_(orig_bias)

            # Reload
            loader = make_loader(benchmark, scale, data_root)

            # LTE-NoCellPhase
            psnr_noc = eval_psnr_for_model(noc_model, loader, device, data_norm, scale)

            tag = 'ID' if scale <= 4 else 'OOD'
            print(f'  x{scale:<3} {tag}:  LTE={psnr_lte:.2f}  '
                  f'LTE(φ=0)={psnr_zero:.2f}  NoC={psnr_noc:.2f}  '
                  f'Δ(φ=0-LTE)={psnr_zero-psnr_lte:+.2f}  '
                  f'Δ(NoC-LTE)={psnr_noc-psnr_lte:+.2f}')

            all_results['LTE'][f'{benchmark}_x{scale}'] = round(psnr_lte, 4)
            all_results['LTE-phase0'][f'{benchmark}_x{scale}'] = round(psnr_zero, 4)
            all_results['LTE-NoCellPhase'][f'{benchmark}_x{scale}'] = round(psnr_noc, 4)

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY: Mean ΔPSNR vs LTE")
    print(f"{'='*65}")
    for condition, label in [('LTE-phase0', 'LTE(φ=0)-LTE'),
                              ('LTE-NoCellPhase', 'LTE-NoCellPhase-LTE')]:
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {output}')


if __name__ == '__main__':
    main()
