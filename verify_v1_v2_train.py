#!/usr/bin/env python3
"""
V1+V2: 100-epoch monitoring run for SC-INR Phase 2.

V1 (frequency monitoring): every 10 epochs, logs omega distribution stats.
V2 (PSNR comparison): reports val PSNR at epochs 50 and 100 vs Phase 1 baseline.

Usage:
  python verify_v1_v2_train.py

This is a PRE-TRAINING verification, not the full 1000-epoch run.
"""

import os, sys, math, json, argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/workspace/SE-INR/Equivariant-ASISR')
import models, datasets, utils


# ── Monitoring hooks ──────────────────────────────────────────────

class OmegaMonitor:
    """Hooks into training to track omega_k(z) distribution."""

    def __init__(self):
        self.history = defaultdict(list)

    def collect(self, model, epoch):
        """Record omega_map statistics from a forward pass."""
        omega = model.omega_map.detach()  # [B, 2K, H, W]
        B, C2K, H, W = omega.shape
        K = C2K // 2

        # Per-frequency mean over batch and spatial dims
        omega_flat = omega.view(B, K, 2, H, W)
        omega_mag = torch.sqrt(omega_flat[:, :, 0]**2 + omega_flat[:, :, 1]**2)  # [B, K, H, W]
        omega_per_k = omega_mag.mean(dim=(0, 2, 3))  # [K]

        self.history['epoch'].append(epoch)
        self.history['mean'].append(omega_per_k.mean().item())
        self.history['std'].append(omega_per_k.std().item())
        self.history['min'].append(omega_per_k.min().item())
        self.history['max'].append(omega_per_k.max().item())

        # Effective number of frequencies (entropy-based)
        # Bin frequencies and compute entropy
        hist = torch.histc(omega_mag.flatten(), bins=50, min=0.0, max=4.0)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        self.history['K_eff'].append(torch.exp(entropy).item())

        # Fraction of frequencies in danger zone (< 0.05)
        frac_low = (omega_per_k < 0.05).float().mean().item()
        self.history['frac_low'].append(frac_low)

        return {
            'epoch': epoch,
            'omega_mean': omega_per_k.mean().item(),
            'omega_std': omega_per_k.std().item(),
            'omega_min': omega_per_k.min().item(),
            'omega_max': omega_per_k.max().item(),
            'K_eff': torch.exp(entropy).item(),
            'frac_low': frac_low,
        }

    def check_health(self):
        """Return warnings if frequency distribution looks unhealthy."""
        warnings = []
        if len(self.history['epoch']) < 2:
            return warnings

        # Check drift: mean omega moving toward 0
        first_mean = self.history['mean'][0]
        last_mean = self.history['mean'][-1]
        delta = (last_mean - first_mean) / first_mean
        if delta < -0.2:
            warnings.append(f"FREQUENCY DRIFT: mean omega dropped {delta*100:.1f}% "
                          f"({first_mean:.3f} -> {last_mean:.3f})")

        # Check diversity: K_eff dropping
        first_K = self.history['K_eff'][0]
        last_K = self.history['K_eff'][-1]
        if last_K < first_K * 0.5:
            warnings.append(f"DIVERSITY LOSS: K_eff dropped from {first_K:.1f} to {last_K:.1f}")

        # Check collapse: fraction near zero increasing
        last_frac = self.history['frac_low'][-1]
        if last_frac > 0.3:
            warnings.append(f"COLLAPSE RISK: {last_frac*100:.1f}% of freqs < 0.05")

        return warnings


# ── Training ──────────────────────────────────────────────────────

def make_dataloader(spec, training=True):
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(
        dataset, batch_size=spec['batch_size'],
        shuffle=training, num_workers=4, pin_memory=True
    )
    return loader


def run_v1_v2(device='cuda:0', epochs=100):
    """Run V1+V2: 100-epoch training with frequency monitoring."""

    # ── Build model ──
    model_spec = {
        'name': 'sc_inr_phase2',
        'args': {
            'encoder_spec': {
                'name': 'edsr-baseline',
                'args': {'no_upsampling': True},
            },
            'imnet_spec': {
                'name': 'mlp',
                'args': {'out_dim': 3, 'hidden_list': [256, 256, 256]},
            },
            'hidden_dim': 256,
            'num_freqs': 128,
            'num_angles': 8,
            'freq_min': 0.1,
            'freq_max': 2.0,
        },
    }
    model = models.make(model_spec).to(device)
    print(f"Phase 2 params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Data ──
    train_spec = {
        'dataset': {
            'name': 'image-folder',
            'args': {
                'root_path': '../Data/DIV2K_train_HR',
                'repeat': 20, 'cache': 'in_memory',
            },
        },
        'wrapper': {
            'name': 'sr-implicit-downsampled',
            'args': {
                'inp_size': 48, 'scale_max': 4,
                'augment': True, 'sample_q': 2304,
            },
        },
        'batch_size': 16,
    }
    val_spec = {
        'dataset': {
            'name': 'image-folder',
            'args': {
                'root_path': '../Data/DIV2K_valid_HR',
                'first_k': 10, 'repeat': 1, 'cache': 'in_memory',
            },
        },
        'wrapper': {
            'name': 'sr-implicit-downsampled',
            'args': {
                'inp_size': 48, 'scale_max': 4,
                'augment': False, 'sample_q': 2304,
            },
        },
        'batch_size': 16,
    }

    train_loader = make_dataloader(train_spec, training=True)
    val_loader = make_dataloader(val_spec, training=False)

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[200, 400, 600, 800], gamma=0.5
    )

    # ── Data norm ──
    inp_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to(device)
    inp_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to(device)
    gt_sub = torch.FloatTensor([0.5]).view(1, 1, -1).to(device)
    gt_div = torch.FloatTensor([0.5]).view(1, 1, -1).to(device)

    monitor = OmegaMonitor()

    # Phase 1 reference val PSNR curve (pre-recorded)
    ref_psnr = {1: 28.88, 50: 30.05, 100: 30.25}  # approximate from Phase 1 log

    print(f"\n{'='*60}")
    print(f"V1+V2: 100-epoch training with frequency monitoring")
    print(f"{'='*60}\n")

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = utils.Averager()

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            for k, v in batch.items():
                batch[k] = v.to(device)

            inp = (batch['inp'] - inp_sub) / inp_div
            pred = model(inp, batch['coord'], batch['cell'])

            gt = (batch['gt'] - gt_sub) / gt_div
            loss = F.l1_loss(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.add(loss.item())

        # Validation
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            val_psnr = utils.Averager()
            val_loss = utils.Averager()

            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Val', leave=False):
                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    inp = (batch['inp'] - inp_sub) / inp_div
                    pred = model(inp, batch['coord'], batch['cell'])

                    pred = pred * gt_div + gt_sub
                    pred.clamp_(0, 1)

                    psnr_val = utils.calc_psnr(pred, batch['gt'])
                    val_psnr.add(psnr_val.item(), inp.shape[0])

                    loss_val = F.l1_loss(pred, batch['gt'])
                    val_loss.add(loss_val.item())

            v_psnr = val_psnr.item()
            v_loss = val_loss.item()

            # Record omega stats
            stats = monitor.collect(model, epoch)
            best_val = max(best_val, v_psnr)

            ref_p = ref_psnr.get(epoch, None)
            ref_str = f"  ref(Phase1): {ref_p:.4f}" if ref_p else ""
            print(f"Epoch {epoch:4d}: train_loss={train_loss.item():.4f}, "
                  f"val_loss={v_loss:.4f}, val_psnr={v_psnr:.4f}{ref_str}, "
                  f"best={best_val:.4f}")
            print(f"  omega: mean={stats['omega_mean']:.4f}, std={stats['omega_std']:.4f}, "
                  f"min={stats['omega_min']:.4f}, max={stats['omega_max']:.4f}, "
                  f"K_eff={stats['K_eff']:.1f}, low%={stats['frac_low']:.3f}")

            # Health check
            warnings = monitor.check_health()
            for w in warnings:
                print(f"  ** WARNING: {w}")

        scheduler.step()

    # ── Final report ──
    print(f"\n{'='*60}")
    print("V1+V2 COMPLETE — 100-epoch summary")
    print(f"{'='*60}")

    print(f"\nBest val PSNR: {best_val:.4f}")
    print(f"Phase 1 reference @ epoch 100: ~30.25")

    h = monitor.history
    print(f"\nOmega distribution evolution:")
    print(f"{'Epoch':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'K_eff':>8} {'low%':>8}")
    for i in range(len(h['epoch'])):
        print(f"{h['epoch'][i]:>6} {h['mean'][i]:>8.4f} {h['std'][i]:>8.4f} "
              f"{h['min'][i]:>8.4f} {h['max'][i]:>8.4f} {h['K_eff'][i]:>8.1f} "
              f"{h['frac_low'][i]:>8.3f}")

    print(f"\nHealth check:")
    warnings = monitor.check_health()
    if warnings:
        for w in warnings:
            print(f"  FAIL: {w}")
    else:
        print(f"  PASS: No collapse or drift detected.")

    # V2 verdict
    print(f"\nV2 VERDICT:")
    if best_val >= 30.2:
        print(f"  PASS: val PSNR ({best_val:.2f}) >= Phase 1 baseline (30.25)")
    elif best_val >= 30.0:
        print(f"  ACCEPTABLE: val PSNR ({best_val:.2f}) slightly below Phase 1")
    else:
        print(f"  FAIL: val PSNR ({best_val:.2f}) significantly below Phase 1")

    return best_val, monitor.history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    run_v1_v2(device, args.epochs)
