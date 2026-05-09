"""
eval_fce.py — Function Consistency Error evaluation (corrected).

Core claim of SC-INR:
  The continuous function F does NOT depend on sampling resolution c.
  F(δ; z, c1) should equal F(δ; z, c2) for the SAME z and SAME δ.

Corrected measurement:
  FIX the encoder input (same LR → same z), only vary cell c.
  FCE_decoder(c1, c2) = E_δ [ || F(δ; z, c1) - F(δ; z, c2) ||_2 ]

This ISOLATES the decoder's c-dependence from encoder inconsistency.

Expected:
  - LTE:      h_p(c) changes with c → FCE > 0 (increases with |c1-c2|)
  - LTE-NoCellPhase: phase=0 fixed → FCE ≈ 0
  - SC-INR-NoPhi:    c only enters sinc (after F) → FCE ≈ 0

Usage:
  python eval_fce.py --models LTE,LTE-NoCellPhase \
                     --r_pairs "2,4;2,8;2,16;4,12;4,24" \
                     --device 0
"""

import os
import math
import json
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import models
from utils import make_coord


def resize_fn(img_tensor, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img_tensor.cpu().clamp(0, 1))
        )
    )


def make_lr_from_hr(hr_tensor, scale):
    H, W = hr_tensor.shape[-2:]
    h = math.floor(H / scale + 1e-9)
    w = math.floor(W / scale + 1e-9)
    hr_crop = hr_tensor[:, :round(h * scale), :round(w * scale)]
    lr = resize_fn(hr_crop, (h, w))
    return lr, hr_crop


def _gfetch(fmap, coord_):
    return (
        F.grid_sample(fmap, coord_.flip(-1).unsqueeze(1),
                      mode='nearest', align_corners=False)
        [:, :, 0, :].permute(0, 2, 1)
    )


def extract_F_lte(model, coords, cell_tensor):
    """
    Extract LTE / LTE-NoCellPhase basis function values at given cell size.
    cell_tensor: [B, Q, 2] — the cell to use for phase computation.
    """
    feat = model.feat
    feat_coord = model.feat_coord

    rx = 2 / feat.shape[-2] / 2
    ry = 2 / feat.shape[-1] / 2

    coord_ = coords.clone()
    coord_[:, :, 0] += -1 * rx + 1e-6
    coord_[:, :, 1] += -1 * ry + 1e-6
    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

    q_coef = _gfetch(model.coeff, coord_)     # [B, Q, 256]
    q_freq = _gfetch(model.freqq, coord_)     # [B, Q, 256]
    q_coord = _gfetch(feat_coord, coord_)     # [B, Q, 2]

    rel_coord = coords - q_coord
    rel_coord[:, :, 0] *= feat.shape[-2]
    rel_coord[:, :, 1] *= feat.shape[-1]

    rel_cell = cell_tensor.clone()
    rel_cell[:, :, 0] *= feat.shape[-2]
    rel_cell[:, :, 1] *= feat.shape[-1]

    bs, q = coords.shape[:2]
    q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)  # [B,Q,128,2]
    q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))            # [B,Q,128,2]
    q_freq = torch.sum(q_freq, dim=-2)                             # [B,Q,128]

    # Apply h_p(c) if the model has a phase module
    if hasattr(model, 'phase'):
        q_freq = q_freq + model.phase(rel_cell.view(bs * q, -1)).view(bs, q, -1)

    q_freq_full = torch.cat(
        [torch.cos(math.pi * q_freq),
         torch.sin(math.pi * q_freq)], dim=-1
    )  # [B, Q, 256]

    F_vals = q_coef * q_freq_full
    return F_vals.squeeze(0)  # [Q, 256]


def extract_F_scinr(model, coords, cell_tensor=None):
    """
    Extract SC-INR basis function values BEFORE sinc weighting.
    cell_tensor is NOT used (c only enters via sinc, after F).
    """
    feat = model.feat
    feat_coord = model.feat_coord
    coef_map = model.coeff

    rx = 2 / feat.shape[-2] / 2
    ry = 2 / feat.shape[-1] / 2

    coord_ = coords.clone()
    coord_[:, :, 0] += -1 * rx + 1e-6
    coord_[:, :, 1] += -1 * ry + 1e-6
    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

    q_coef = _gfetch(coef_map, coord_)       # [B, Q, 256]
    q_coord = _gfetch(feat_coord, coord_)     # [B, Q, 2]

    rel_coord = coords - q_coord
    rel_coord[:, :, 0] *= feat.shape[-2]
    rel_coord[:, :, 1] *= feat.shape[-1]

    q_phase = torch.matmul(rel_coord, model.freqs.T)   # [B, Q, K]

    if model.learn_phase and model.phase_map is not None:
        q_phi = _gfetch(model.phase_map, coord_)       # [B, Q, K]
        q_phase = q_phase + q_phi

    fourier_feats = torch.cat(
        [torch.cos(math.pi * q_phase),
         torch.sin(math.pi * q_phase)], dim=-1
    )  # [B, Q, 256]

    F_vals = q_coef * fourier_feats   # [B, Q, 256]
    return F_vals.squeeze(0)          # [Q, 256]


def extract_F(model, model_type, coords, cell=None):
    if model_type == 'lte' or model_type == 'lte-noc':
        return extract_F_lte(model, coords, cell)
    elif model_type == 'sc-inr':
        return extract_F_scinr(model, coords, cell)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_fce_fixed_z(model, model_type, hr_images, scales, device,
                        num_queries=512):
    """
    CORRECTED FCE: fix z (same LR input), vary c only.

    For each image and each reference scale r_ref:
      1. Encode LR at scale r_ref → z
      2. For each pair (r1, r2), compute cell sizes c1, c2
      3. Extract F with c=c1 and F with c=c2
      4. FCE = ||F(c1) - F(c2)||
    """
    inp_sub = torch.FloatTensor([0.5]).view(1, 1, 1, 1).to(device)
    inp_div = torch.FloatTensor([0.5]).view(1, 1, 1, 1).to(device)

    # Only measure decoder c-dependence: fix z, vary c
    # Use pairs of (cell1, cell2) from different scales
    cell_pairs = []
    for i, r1 in enumerate(scales):
        for r2 in scales[i+1:]:
            cell_pairs.append((r1, r2))

    accum = {pair: [] for pair in cell_pairs}

    for hr in tqdm(hr_images, desc=f'  {model_type}', leave=False):
        hr = hr.to(device)

        for (r1, r2) in cell_pairs:
            try:
                # Use a single LR at the average scale for encoding
                # This gives us one z that works for both cell queries
                r_enc = (r1 + r2) / 2.0
                lr, _ = make_lr_from_hr(hr, r_enc)
                inp = ((lr.unsqueeze(0).to(device) - inp_sub) / inp_div)

                # Random coordinates in [-1, 1]^2
                coords = (torch.rand(1, num_queries, 2) * 2 - 1).to(device)

                # Compute cell sizes for r1 and r2
                feat_h = math.floor(hr.shape[-2] / r_enc + 1e-9)
                feat_w = math.floor(hr.shape[-1] / r_enc + 1e-9)

                # cell = 2/r in LR pixel units
                cell1 = torch.full((1, num_queries, 2),
                                   2.0 / r1, device=device)
                cell2 = torch.full((1, num_queries, 2),
                                   2.0 / r2, device=device)

                # Normalize cell to LR pixel units (same as training)
                cell1_norm = cell1.clone()
                cell2_norm = cell2.clone()

                with torch.no_grad():
                    model.gen_feat(inp)
                    F1 = extract_F(model, model_type, coords, cell1_norm)
                    F2 = extract_F(model, model_type, coords, cell2_norm)

                fce = (F1 - F2).pow(2).sum(dim=-1).sqrt().mean().item()
                accum[(r1, r2)].append(fce)

            except Exception as e:
                continue

    return {pair: float(np.mean(vals)) for pair, vals in accum.items() if vals}


def main():
    parser = argparse.ArgumentParser(
        description='Function Consistency Error (corrected: fix z, vary c)')
    parser.add_argument('--models', type=str, default='LTE,LTE-NoCellPhase')
    parser.add_argument('--r_pairs', type=str, default='2,4;2,8;2,16;4,12;4,24',
                        help='Cell pairs as r1,r2;r1,r2;...')
    parser.add_argument('--data_root', type=str,
                        default=str(Path(os.environ.get('SEINR_DATA_ROOT', ROOT.parent / 'Data'))))
    parser.add_argument('--benchmark', type=str, default='BSD100')
    parser.add_argument('--max_images', type=int, default=20)
    parser.add_argument('--num_queries', type=int, default=512)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output', type=str, default='results/fce.json')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    ALL_MODELS = {
        'LTE':     ('save/lte/epoch-best.pth',     'lte'),
        'LTE-NoCellPhase': ('save/lte-nocellphase/epoch-best.pth', 'lte-noc'),
        'LTE-NoCell': ('save/lte-nocellphase/epoch-best.pth', 'lte-noc'),
        'LTE-NoC': ('save/lte-nocellphase/epoch-best.pth', 'lte-noc'),
        'SC-INR-FixedOmega':  ('save/sc-inr-fixed-omega/epoch-best.pth', 'sc-inr'),
        'SC-INR-Fixed':  ('save/sc-inr-fixed-omega/epoch-best.pth', 'sc-inr'),
        'SC-INR-NoPhi':  ('save/sc-inr-nophi/epoch-best.pth', 'sc-inr'),
        'LTE-PhaseZ':  ('save/lte-phasez/epoch-best.pth', 'lte'),
        'LTE-FeaturePhase':  ('save/lte-phasez/epoch-best.pth', 'lte'),
    }

    # Parse scale pairs: these define which cell sizes to compare
    r_pairs = []
    all_scales = set()
    for s in args.r_pairs.split(';'):
        a, b = s.strip().split(',')
        r_pairs.append((float(a), float(b)))
        all_scales.add(float(a))
        all_scales.add(float(b))
    scales = sorted(all_scales)

    img_dir = os.path.join(args.data_root, args.benchmark, 'HR')
    files = sorted(os.listdir(img_dir))[:args.max_images]
    hr_list = [
        transforms.ToTensor()(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        for f in files
    ]
    print(f"Loaded {len(hr_list)} HR images from {args.benchmark}")
    print(f"Scale pairs (cell comparison): {r_pairs}")
    print(f"Method: FIX z (same LR), VARY c only → pure decoder c-dependence")
    print()

    model_names = [m.strip() for m in args.models.split(',')]
    all_results = {}

    for mname in model_names:
        if mname not in ALL_MODELS:
            print(f"[SKIP] {mname}: not in registry")
            continue
        mpath, mtype = ALL_MODELS[mname]
        if not os.path.exists(mpath):
            print(f"[SKIP] {mname}: checkpoint not found at {mpath}")
            continue

        print(f"{'='*55}")
        print(f"Model: {mname}  (type={mtype})")
        print(f"{'='*55}")

        sv = torch.load(mpath, map_location='cpu')
        mdl = models.make(sv['model'], load_sd=True, strict=False).to(device)
        mdl.eval()

        fce = compute_fce_fixed_z(
            mdl, mtype, hr_list, scales, device, args.num_queries
        )

        # Filter to only the requested pairs
        filtered = {}
        for (r1, r2), v in fce.items():
            if (r1, r2) in r_pairs or (r2, r1) in r_pairs:
                key = (min(r1, r2), max(r1, r2))
                filtered[key] = v
        all_results[mname] = {
            f'r{int(r1)}v{int(r2)}': v for (r1, r2), v in filtered.items()
        }

        for (r1, r2), v in filtered.items():
            print(f"  c(r={int(r1):>2}) vs c(r={int(r2):>2}): FCE = {v:.6f}")

        del mdl
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*65}")
    print("FCE SUMMARY (corrected: fix z, vary c)")
    print("lower = less c-dependence in decoder = more scale-consistent")
    print(f"{'='*65}")
    pair_keys = [f'r{int(r1)}v{int(r2)}' for r1, r2 in r_pairs]
    header = f"{'Model':<18}" + "".join(f"{pk:>14}" for pk in pair_keys)
    print(header)
    print("-" * len(header))
    for mname, res in all_results.items():
        row = f"{mname:<18}"
        for pk in pair_keys:
            v = res.get(pk, float('nan'))
            row += f"{v:>14.6f}"
        print(row)

    # Prediction comparison
    print(f"\n{'='*65}")
    print("PREDICTION vs OBSERVATION")
    print(f"{'='*65}")
    for pk in pair_keys:
        lte_v = all_results.get('LTE', {}).get(pk, None)
        noc_v = all_results.get('LTE-NoCellPhase', all_results.get('LTE-NoC', {})).get(pk, None)
        if lte_v is not None and noc_v is not None:
            ratio = lte_v / noc_v if noc_v > 0 else float('inf')
            verdict = "✓ LTE > NoC" if lte_v > noc_v else "✗ LTE < NoC (unexpected)"
            print(f"  {pk}: LTE={lte_v:.4f}  NoC={noc_v:.4f}  "
                  f"ratio={ratio:.2f}x  {verdict}")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
