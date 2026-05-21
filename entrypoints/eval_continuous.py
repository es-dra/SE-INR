#!/usr/bin/env python3
"""
Continuous PSNR curve evaluation using benchmark framework.
Evaluates ASISR models on benchmark datasets at scales r in [1, 30], step 0.5.
Output: comparison plot and JSON of results.

Usage:
    python entrypoints/eval_continuous.py --dataset set5     # Set5 only (default)
    python entrypoints/eval_continuous.py --dataset bsd100   # BSD100 only
    python entrypoints/eval_continuous.py --dataset all      # Both datasets
"""
import os
import sys
import math
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import models
import utils
from scripts.analysis.model_registry import MODEL_ALIASES


DATA_ROOT = Path(os.environ.get('SEINR_DATA_ROOT', ROOT.parent / 'Data'))
DATASETS = {
    'set5': str(DATA_ROOT / 'Set5' / 'HR'),
    'bsd100': str(DATA_ROOT / 'BSD100' / 'HR'),
    'set14': str(DATA_ROOT / 'Set14' / 'HR'),
    'urban100': str(DATA_ROOT / 'Urban100' / 'HR'),
}

ALL_DATASETS = list(DATASETS.keys())

class ContinuousBenchmark:
    """Evaluate models at continuous scale factors using benchmark-style evaluation."""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        ckpt = torch.load(model_path, map_location='cpu')
        self.model = models.make(ckpt['model'], load_sd=True, strict=False).to(device)
        self.model.eval()

    def eval_image_at_scale(self, img_path, scale):
        """Evaluate one image at a specific scale factor using benchmark-style evaluation.

        Matches SRImplicitDownsampled pipeline exactly:
        1. Crop HR to round(h_lr*s) x round(w_lr*s) first
        2. Create LR via BICUBIC downsample of cropped HR
        3. Predict at target resolution using cropped HR as GT
        """
        img_hr_pil = Image.open(img_path).convert('RGB')

        w_hr, h_hr = img_hr_pil.size  # PIL returns (W, H)
        h_lr = max(1, int(math.floor(h_hr / scale + 1e-9)))
        w_lr = max(1, int(math.floor(w_hr / scale + 1e-9)))
        target_h = int(round(h_lr * scale))
        target_w = int(round(w_lr * scale))

        # Crop HR first (matching SRImplicitDownsampled: img[:, :round(h_lr*s), :round(w_lr*s)])
        img_hr_cropped_pil = img_hr_pil.crop((0, 0, target_w, target_h))

        # Create LR from cropped HR (matching resize_fn in wrappers.py)
        img_lr_pil = img_hr_cropped_pil.resize((w_lr, h_lr), Image.BICUBIC)
        img_lr = transforms.ToTensor()(img_lr_pil).unsqueeze(0).to(self.device)

        # GT is the cropped HR (exact alignment with LR)
        img_hr_cropped = transforms.ToTensor()(img_hr_cropped_pil).to(self.device)

        # Normalization (same as benchmark / eval_full.py)
        inp_sub = torch.tensor([0.5]).view(1, -1, 1, 1).to(self.device)
        inp_div = torch.tensor([0.5]).view(1, -1, 1, 1).to(self.device)
        gt_sub = torch.tensor([0.5]).view(1, 1, -1).to(self.device)
        gt_div = torch.tensor([0.5]).view(1, 1, -1).to(self.device)

        inp = (img_lr - inp_sub) / inp_div

        # Create coordinate grid at target resolution
        coord = utils.make_coord([target_h, target_w]).unsqueeze(0).to(self.device)
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / target_h
        cell[:, :, 1] *= 2 / target_w

        # Predict in batches using normalized input
        with torch.no_grad():
            self.model.gen_feat(inp)
            n = coord.shape[1]
            preds = []
            for ql in range(0, n, 50000):
                qr = min(ql + 50000, n)
                pred = self.model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
                preds.append(pred)
            pred = torch.cat(preds, dim=1)

        # Denormalize
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        # Reshape to image (matching eval_full.py: pred.view(*shape).permute(0,3,1,2))
        shape = [1, target_h, target_w, 3]
        pred_img = pred.view(*shape).permute(0, 3, 1, 2).contiguous()

        # GT: cropped HR, reshaped to match (matching eval_full.py: gt.view(*shape).permute(0,3,1,2))
        gt = img_hr_cropped.unsqueeze(0)

        # Clip pred to gt size for safety (matching eval_full.py: pred[..., :gt.shape[-2], :gt.shape[-1]])
        pred_img = pred_img[..., :gt.shape[-2], :gt.shape[-1]]

        # Use benchmark-style calc_psnr for proper grayscale + shave handling
        psnr = utils.calc_psnr(pred_img, gt, dataset='benchmark', scale=int(round(scale)), rgb_range=1)
        return float(psnr)

    def eval_dataset(self, dataset_dir, scales):
        """Evaluate on a dataset for given scales."""
        image_names = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.png')])

        results = {}
        for scale in scales:
            psnr_values = []
            for img_name in tqdm(image_names, desc=f'r={scale:.1f}', leave=False):
                img_path = os.path.join(dataset_dir, img_name)
                try:
                    psnr = self.eval_image_at_scale(img_path, scale)
                    psnr_values.append(psnr)
                except Exception as e:
                    print(f'Error at scale={scale:.1f}, img={img_name}: {e}')
                    psnr_values.append(float('nan'))
            avg_psnr = np.nanmean(psnr_values)
            results[f'{scale:.1f}'] = round(avg_psnr, 4)
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Continuous PSNR evaluation. 默认读取 canonical checkpoint root。'
    )
    parser.add_argument('--dataset', type=str, default='set5',
                        choices=['set5', 'bsd100', 'set14', 'urban100', 'all'],
                        help='Dataset to evaluate on (default: set5)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated models to evaluate. Default: all available')
    parser.add_argument(
        '--save_root',
        type=str,
        default='artifacts/checkpoints/seed1',
        help='Checkpoint root. Formal runs should prefer artifacts/checkpoints/seedN, not legacy save/.'
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    if args.dataset == 'all':
        target_datasets = ALL_DATASETS
    else:
        target_datasets = [args.dataset]

    save_root = args.save_root.rstrip('/')
    MODELS = {
        'LIIF': f'{save_root}/liif/epoch-best.pth',
        'LIIF-EQ': f'{save_root}/liif-eq/epoch-best.pth',
        'LTE': f'{save_root}/lte/epoch-best.pth',
        'LTE-NoCellPhase': f'{save_root}/lte-nocellphase/epoch-best.pth',
        'LTE-EQ': f'{save_root}/lte-eq/epoch-best.pth',
        'LTE-PhaseZ': f'{save_root}/lte-phasez/epoch-best.pth',
        'SC-INR-FixedOmega': f'{save_root}/sc-inr-fixed-omega/epoch-best.pth',
        'SC-INR-NoPhi': f'{save_root}/sc-inr-nophi/epoch-best.pth',
        'SC-INR-NoPhi-Signed': f'{save_root}/sc-inr-nophi-signed/epoch-best.pth',
        'SC-INR': f'{save_root}/sc-inr/epoch-best.pth',
        'SC-INR-NoSinc': f'{save_root}/sc-inr-nosinc/epoch-best.pth',
    }
    if args.models:
        model_names = [MODEL_ALIASES.get(m.strip(), m.strip()) for m in args.models.split(',')]
        MODELS = {k: v for k, v in MODELS.items() if k in model_names}
    available = {k: v for k, v in MODELS.items() if os.path.exists(v)}
    if not available:
        print('No models found!')
        return

    scales = np.arange(1.0, 30.5, 0.5).tolist()
    all_results = {}

    for model_name, model_path in available.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")

        evaluator = ContinuousBenchmark(model_path, device)
        for ds_name in target_datasets:
            ds_dir = DATASETS[ds_name]
            results = evaluator.eval_dataset(ds_dir, scales)
            all_results[f"{model_name}_{ds_name}"] = results

        print(f"  Completed {len(scales)} scale points on {len(target_datasets)} dataset(s)")

    # Save JSON per dataset — merge with existing data if present
    for ds_name in target_datasets:
        ds_results = {}
        json_path = os.path.join('results', 'continuous', f'{ds_name}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                ds_results = json.load(f)
        for model_name in available:
            ds_results[model_name] = all_results[f"{model_name}_{ds_name}"]
        with open(json_path, 'w') as f:
            json.dump(ds_results, f, indent=2)
        print(f"Saved {json_path}")

    # Generate plot per dataset
    for ds_name in target_datasets:
        plt.figure(figsize=(12, 8))
        colors = {'LIIF': '#1f77b4', 'LIIF-EQ': '#17becf', 'LTE': '#ff7f0e',
                  'LTE-NoCellPhase': '#2ca02c', 'LTE-EQ': '#bcbd22',
                  'LTE-PhaseZ': '#9467bd', 'SC-INR-FixedOmega': '#d62728',
                  'SC-INR-NoPhi': '#8c564b', 'SC-INR-NoPhi-Signed': '#aa3377',
                  'SC-INR': '#b22222', 'SC-INR-NoSinc': '#666666'}
        markers = {'LIIF': 'o', 'LIIF-EQ': 'v', 'LTE': 's', 'LTE-NoCellPhase': '^',
                   'LTE-EQ': '<', 'LTE-PhaseZ': '>', 'SC-INR-FixedOmega': 'D',
                   'SC-INR-NoPhi': 'p', 'SC-INR-NoPhi-Signed': 'X',
                   'SC-INR': '*', 'SC-INR-NoSinc': 'h'}

        ds_results = {k: v for k, v in all_results.items() if k.endswith(f'_{ds_name}')}
        for model_name in available:
            key = f"{model_name}_{ds_name}"
            if key not in ds_results:
                continue
            xs = sorted([float(k) for k in ds_results[key].keys()])
            ys = [ds_results[key][f'{x:.1f}'] for x in xs]
            plt.plot(xs, ys, label=model_name, color=colors.get(model_name),
                     marker=markers.get(model_name, 'o'), markevery=4, markersize=4, linewidth=1.5)

        plt.xlabel('Scale Factor (r)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title(f'Continuous PSNR Curve on {ds_name.upper()}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(0.5, 31)

        # Mark ID vs OOD boundary
        plt.axvline(x=4, color='gray', linestyle='--', alpha=0.5)
        plt.axvspan(1, 4, alpha=0.08, color='blue')
        plt.axvspan(4, 30.5, alpha=0.08, color='red')
        plt.text(2.5, 40, 'ID', fontsize=10, ha='center', color='blue')
        plt.text(17, 40, 'OOD', fontsize=10, ha='center', color='red')

        plt.tight_layout()
        png_path = os.path.join('results', 'figures', f'continuous_{ds_name}.png')
        plt.savefig(png_path, dpi=150)
        print(f"Plot saved to {png_path}")

        # Summary table
        print(f"\n{'='*80}")
        print(f"Selected Scale Results on {ds_name.upper()} (PSNR dB)")
        print(f"{'='*80}")
        selected = [2, 3, 4, 6, 8, 12, 16, 24, 30]
        header = f"{'Scale':>8}"
        for name in available:
            header += f"{name:>12}"
        print(header)
        print("-" * len(header))
        for s in selected:
            row = f"x{s:>7.1f}"
            for name in available:
                v = ds_results[f"{name}_{ds_name}"].get(f'{float(s):.1f}', None)
                row += f"{v:>12.2f}" if v is not None else f"{'N/A':>12}"
            print(row)


if __name__ == '__main__':
    main()
