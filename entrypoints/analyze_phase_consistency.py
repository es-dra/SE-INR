"""
Analyze whether SC-INR's phase_conv(z) implicitly encodes scale information.

Hypothesis:
  Since SC-INR uses fixed frequencies, phase_conv must adapt to both image content
  AND scale. Different LR scales produce statistically different z, and phase_conv(z)
  may learn to read scale from z, effectively replacing explicit h_p(c) with implicit
  scale encoding.

Test:
  For the same HR patch, generate LR at r1 and r2, encode to get z1 and z2.
  Compare phase_conv(z1) vs phase_conv(z2) at the same spatial location.
  If |Δφ_k| correlates with ω_k (higher freq → larger phase diff), it suggests
  phase_conv is doing implicit scale compensation.
"""

import os, sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import models
from utils import make_coord


def make_lr(hr_tensor, scale):
    """Bicubic downsample HR to given scale."""
    h, w = hr_tensor.shape[-2:]
    hl = max(1, int(math.floor(h / scale + 1e-9)))
    wl = max(1, int(math.floor(w / scale + 1e-9)))
    return transforms.ToTensor()(
        transforms.Resize((hl, wl), transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(hr_tensor.cpu().clamp(0, 1))
        )
    )


def analyze_phase_shift(model, hr_tensor, r1, r2, device):
    """
    Compare phase_conv outputs for two different LR scales of the same HR image.

    Returns:
        phase_diff: [K] mean absolute phase difference
        freqs: [K, 2] frequency basis
    """
    inp_sub = torch.tensor([0.5]).view(1, -1, 1, 1).to(device)
    inp_div = torch.tensor([0.5]).view(1, -1, 1, 1).to(device)

    lr1 = make_lr(hr_tensor, r1).unsqueeze(0).to(device)
    lr2 = make_lr(hr_tensor, r2).unsqueeze(0).to(device)

    inp1 = (lr1 - inp_sub) / inp_div
    inp2 = (lr2 - inp_sub) / inp_div

    with torch.no_grad():
        model.gen_feat(inp1)
        pmap1 = model.phase_map.clone()  # [1, K, H1, W1]
        coord1 = model.feat_coord.clone()

        model.gen_feat(inp2)
        pmap2 = model.phase_map.clone()  # [1, K, H2, W2]
        coord2 = model.feat_coord.clone()

    # Sample random query points in normalized [-1, 1] space
    # These correspond to the SAME physical locations in both LR grids
    n_queries = 256
    queries = (torch.rand(1, n_queries, 2) * 2 - 1).to(device)

    # Sample phase from both feature maps at same normalized coords
    q_phase1 = F.grid_sample(pmap1, queries.flip(-1).unsqueeze(1),
                              mode='bilinear', align_corners=False
                              )[:, :, 0, :].permute(0, 2, 1)  # [1, Q, K]
    q_phase2 = F.grid_sample(pmap2, queries.flip(-1).unsqueeze(1),
                              mode='bilinear', align_corners=False
                              )[:, :, 0, :].permute(0, 2, 1)  # [1, Q, K]

    # Mean absolute phase difference per frequency
    phase_diff = (q_phase1 - q_phase2).abs().mean(dim=0).mean(dim=0)  # [K]
    phase_diff_std = (q_phase1 - q_phase2).abs().mean(dim=0).std(dim=0)  # [K]

    return phase_diff.cpu(), phase_diff_std.cpu(), model.freqs.cpu()


def main():
    device = 'cuda:0'

    # Load SC-INR model
    ckpt = torch.load('save/sc-inr-fixed/epoch-best.pth', map_location='cpu')
    model = models.make(ckpt['model'], load_sd=True, strict=False).to(device)
    model.eval()

    # Load test images
    data_root = Path(os.environ.get('SEINR_DATA_ROOT', ROOT.parent / 'Data'))
    img_dir = str(data_root / 'BSD100' / 'HR')
    files = sorted(os.listdir(img_dir))[:20]
    hr_images = []
    for fn in files:
        hr = transforms.ToTensor()(Image.open(os.path.join(img_dir, fn)).convert('RGB'))
        hr_images.append(hr)
    print(f'Loaded {len(hr_images)} images')

    # Scale pairs to test
    pairs = [(2, 4), (2, 8), (2, 16), (4, 12), (4, 24), (2, 3)]

    K = model.num_freqs
    all_diffs = {pair: [] for pair in pairs}

    for r1, r2 in tqdm(pairs, desc='Scale pairs'):
        for hr in tqdm(hr_images, desc=f'  r{r1}vs{r2}', leave=False):
            try:
                diff, std, freqs = analyze_phase_shift(model, hr, r1, r2, device)
                all_diffs[(r1, r2)].append(diff)
            except Exception as e:
                continue

    # Aggregate results
    print(f'\n{"="*70}')
    print('Phase shift per frequency band (averaged over images)')
    print(f'{"="*70}')

    freqs = model.freqs.cpu()
    freq_mags = (freqs[:, 0]**2 + freqs[:, 1]**2).sqrt()  # [K]

    # Sort frequencies by magnitude for analysis
    sorted_idx = freq_mags.argsort()

    # Bin into low/mid/high frequency bands
    K_per_band = K // 3
    bands = {
        'low': sorted_idx[:K_per_band],
        'mid': sorted_idx[K_per_band:2*K_per_band],
        'high': sorted_idx[2*K_per_band:],
    }

    print(f'\nFreq range: [{freq_mags.min():.3f}, {freq_mags.max():.3f}]')
    print(f'Band boundaries: low<={freq_mags[bands["low"]].max():.3f}, '
          f'mid<={freq_mags[bands["mid"]].max():.3f}, '
          f'high<={freq_mags[bands["high"]].max():.3f}')
    print()

    # Summary table
    header = f'{"Pair":<12} {"Low Δφ":>10} {"Mid Δφ":>10} {"High Δφ":>10} {"Corr(ω,Δφ)":>12}'
    print(header)
    print('-' * len(header))

    for (r1, r2), diffs in all_diffs.items():
        if not diffs:
            continue
        # Average over images
        avg_diff = torch.stack(diffs).mean(dim=0)  # [K]

        band_vals = {}
        for band_name, idx in bands.items():
            band_vals[band_name] = avg_diff[idx].mean().item()

        # Correlation between frequency magnitude and phase difference
        corr = np.corrcoef(freq_mags.numpy(), avg_diff.numpy())[0, 1]

        print(f'r{r1} vs r{r2:<4}  {band_vals["low"]:>10.4f}  '
              f'{band_vals["mid"]:>10.4f}  {band_vals["high"]:>10.4f}  '
              f'{corr:>12.4f}')

    # Interpretation
    print(f'\n{"="*70}')
    print('INTERPRETATION')
    print(f'{"="*70}')
    print('If High_Δφ > Low_Δφ and Corr(ω,Δφ) > 0:')
    print('  → phase_conv is doing implicit scale compensation')
    print('  → higher frequencies get larger phase shifts between scales')
    print('  → c-dependence has been moved from explicit h_p(c) to implicit phase_conv(z)')
    print()
    print('If Δφ ≈ constant across bands and Corr(ω,Δφ) ≈ 0:')
    print('  → phase_conv captures only content, not scale')
    print('  → phase differences are from image content, not scale encoding')


if __name__ == '__main__':
    main()
