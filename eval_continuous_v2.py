#!/usr/bin/env python3
"""
Continuous PSNR curve evaluation using benchmark framework.
Evaluates LIIF, LTE, LTE-NoC on Set5/BSD100 at scales r ∈ [1, 30], step 0.5.
Output: comparison plot and JSON of results.

Usage:
    python eval_continuous_v2.py --dataset set5     # Set5 only (default)
    python eval_continuous_v2.py --dataset bsd100   # BSD100 only
    python eval_continuous_v2.py --dataset all      # Both datasets
"""
import os
import sys
import math
import json
import argparse

os.chdir('/workspace/SE-INR/Equivariant-ASISR')
sys.path.insert(0, '.')

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


DATASETS = {
    'set5': '/workspace/SE-INR/Data/Set5/HR',
    'bsd100': '/workspace/SE-INR/Data/BSD100/HR',
    'set14': '/workspace/SE-INR/Data/Set14/HR',
    'urban100': '/workspace/SE-INR/Data/Urban100/HR',
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
        """Evaluate one image at a specific scale factor using benchmark-style evaluation."""
        img_hr_pil = Image.open(img_path).convert('RGB')

        w_hr, h_hr = img_hr_pil.size  # PIL returns (W, H)
        h_lr = max(1, int(math.floor(h_hr / scale + 1e-9)))
        w_lr = max(1, int(math.floor(w_hr / scale + 1e-9)))
        target_h = int(round(h_lr * scale))
        target_w = int(round(w_lr * scale))

        # Create LR via direct resize (benchmark style)
        img_lr_pil = img_hr_pil.resize((w_lr, h_lr), Image.BICUBIC)
        img_lr = transforms.ToTensor()(img_lr_pil).unsqueeze(0).to(self.device)

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

        # Reshape to image using correct ordering (same as eval_full.py)
        # s = sqrt(N / (h_lr * w_lr)) where s ≈ scale
        # But due to rounding in target_h/target_w, s is approximate
        # Use s = scale directly for exact scaling
        s = scale
        shape = [1, round(h_lr * s), round(w_lr * s), 3]
        pred_img = pred.view(*shape).permute(0, 3, 1, 2).contiguous()

        # GT image - use actual pred size (not target_h/target_w which may be rounded)
        img_hr_tensor = transforms.ToTensor()(img_hr_pil).to(self.device)
        actual_h, actual_w = pred_img.shape[-2], pred_img.shape[-1]
        gt = img_hr_tensor[:, :actual_h, :actual_w].unsqueeze(0)

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
    parser = argparse.ArgumentParser(description='Continuous PSNR evaluation')
    parser.add_argument('--dataset', type=str, default='set5',
                        choices=['set5', 'bsd100', 'set14', 'urban100', 'all'],
                        help='Dataset to evaluate on (default: set5)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    if args.dataset == 'all':
        target_datasets = ALL_DATASETS
    else:
        target_datasets = [args.dataset]

    MODELS = {
        'LIIF': 'save/edsr-baseline-liif/epoch-best.pth',
        'LTE': 'save/edsr-baseline-lte/epoch-best.pth',
        'LTE-NoC': 'save/edsr-baseline-lte-noc/epoch-best.pth',
    }

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

    # Save JSON per dataset
    for ds_name in target_datasets:
        ds_results = {}
        for model_name in available:
            ds_results[model_name] = all_results[f"{model_name}_{ds_name}"]
        json_path = f'eval_continuous_{ds_name}.json'
        with open(json_path, 'w') as f:
            json.dump(ds_results, f, indent=2)
        print(f"Saved {json_path}")

    # Generate plot per dataset
    for ds_name in target_datasets:
        plt.figure(figsize=(12, 8))
        colors = {'LIIF': '#1f77b4', 'LTE': '#ff7f0e', 'LTE-NoC': '#2ca02c'}
        markers = {'LIIF': 'o', 'LTE': 's', 'LTE-NoC': '^'}

        ds_results = {k: v for k, v in all_results.items() if k.endswith(f'_{ds_name}')}
        for model_name in available:
            key = f"{model_name}_{ds_name}"
            if key not in ds_results:
                continue
            xs = sorted([float(k) for k in ds_results[key].keys()])
            ys = [ds_results[key][f'{x:.1f}'] for x in xs]
            plt.plot(xs, ys, label=model_name, color=colors[model_name],
                     marker=markers[model_name], markevery=4, markersize=4, linewidth=1.5)

        plt.xlabel('Scale Factor (r)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title(f'Continuous PSNR Curve on {ds_name.upper()}\nLIIF vs LTE vs LTE-NoC', fontsize=14)
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
        png_path = f'eval_continuous_{ds_name}.png'
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