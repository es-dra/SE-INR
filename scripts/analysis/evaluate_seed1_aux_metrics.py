#!/usr/bin/env python3
"""Auxiliary metrics and visualizations for seed1 ASISR checkpoints.

The goal is to complement PSNR with scale-focused diagnostics that are directly
useful for the current SC-INR paper narrative:

1. benchmark-style quality metrics at selected integer scales;
2. region-aware errors on edge / texture / flat pixels;
3. high-frequency and FFT-band error summaries;
4. same-LR cross-scale observation consistency;
5. high-texture crop visualizations.

Outputs are written under results/analysis/seed1_aux_metrics/ by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as skimage_ssim
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parent / "Data"
sys.path.insert(0, str(ROOT))

import models
import utils

DATASETS = {
    "set5": DATA_ROOT / "Set5" / "HR",
    "set14": DATA_ROOT / "Set14" / "HR",
    "bsd100": DATA_ROOT / "BSD100" / "HR",
    "urban100": DATA_ROOT / "Urban100" / "HR",
}

MODEL_PATHS = {
    "LIIF": ROOT / "save" / "liif" / "epoch-best.pth",
    "LTE": ROOT / "save" / "lte" / "epoch-best.pth",
    "LTE-NoCellPhase": ROOT / "save" / "lte-nocellphase" / "epoch-best.pth",
    "LTE-NoCell": ROOT / "save" / "lte-nocellphase" / "epoch-best.pth",
    "LTE-NoC": ROOT / "save" / "lte-nocellphase" / "epoch-best.pth",
    "LTE-PhaseZ": ROOT / "save" / "lte-phasez" / "epoch-best.pth",
    "LTE-FeaturePhase": ROOT / "save" / "lte-phasez" / "epoch-best.pth",
    "SC-INR-FixedOmega": ROOT / "save" / "sc-inr-fixed-omega" / "epoch-best.pth",
    "SC-INR-Fixed": ROOT / "save" / "sc-inr-fixed-omega" / "epoch-best.pth",
    "SC-INR-NoPhi": ROOT / "save" / "sc-inr-nophi" / "epoch-best.pth",
    "SC-INR-Adaptive": ROOT / "save" / "sc-inr-nophi" / "epoch-best.pth",
    "SC-INR-NoPhi-Signed": ROOT / "save" / "sc-inr-nophi-signed" / "epoch-best.pth",
    "SC-INR-Signed": ROOT / "save" / "sc-inr-nophi-signed" / "epoch-best.pth",
    "SC-INR": ROOT / "save" / "sc-inr" / "epoch-best.pth",
    "SC-INR+PhiZ": ROOT / "save" / "sc-inr" / "epoch-best.pth",
    "SC-INR-NoSinc": ROOT / "save" / "sc-inr-nosinc" / "epoch-best.pth",
    "LIIF-EQ": ROOT / "save" / "liif-eq" / "epoch-best.pth",
    "LTE-EQ": ROOT / "save" / "lte-eq" / "epoch-best.pth",
}

STYLE = {
    "Bicubic": "#8c8c8c",
    "LIIF": "#4C72B0",
    "LIIF-EQ": "#64B5CD",
    "LTE": "#DD8452",
    "LTE-NoCellPhase": "#55A868",
    "LTE-EQ": "#DDAA33",
    "LTE-NoCell": "#55A868",
    "LTE-PhaseZ": "#8172B2",
    "LTE-FeaturePhase": "#8172B2",
    "SC-INR-FixedOmega": "#C44E52",
    "SC-INR-Fixed": "#C44E52",
    "SC-INR-NoPhi": "#8B0000",
    "SC-INR-Adaptive": "#8B0000",
    "SC-INR-NoPhi-Signed": "#AA3377",
    "SC-INR-Signed": "#AA3377",
    "SC-INR": "#B22222",
    "SC-INR+PhiZ": "#B22222",
    "SC-INR-NoSinc": "#666666",
}


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_scales(text: str) -> List[int]:
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


def parse_pairs(text: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        src = int(float(a.strip()))
        obs = int(float(b.strip()))
        if src <= obs:
            raise ValueError(f"source scale must be larger than observation scale: {chunk}")
        pairs.append((src, obs))
    return pairs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(dataset: str, max_images: int | None) -> List[Path]:
    root = DATASETS[dataset]
    paths = sorted([p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def pil_to_tensor(img: Image.Image, device: torch.device | str) -> torch.Tensor:
    return transforms.ToTensor()(img.convert("RGB")).unsqueeze(0).to(device)


def resize_tensor(img: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(img, size=size_hw, mode="bicubic", align_corners=False).clamp(0, 1)


def make_lr_hr(img_path: Path, scale: float, device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor]:
    img_hr_pil = Image.open(img_path).convert("RGB")
    w_hr, h_hr = img_hr_pil.size
    h_lr = max(1, int(math.floor(h_hr / scale + 1e-9)))
    w_lr = max(1, int(math.floor(w_hr / scale + 1e-9)))
    target_h = int(round(h_lr * scale))
    target_w = int(round(w_lr * scale))
    hr_crop = img_hr_pil.crop((0, 0, target_w, target_h))
    lr = hr_crop.resize((w_lr, h_lr), Image.BICUBIC)
    return pil_to_tensor(lr, device), pil_to_tensor(hr_crop, device)


def load_model(model_name: str, device: torch.device | str):
    path = MODEL_PATHS[model_name]
    ckpt = torch.load(path, map_location="cpu")
    model = models.make(ckpt["model"], load_sd=True, strict=False).to(device)
    model.eval()
    return model


def predict_image(model, lr: torch.Tensor, target_h: int, target_w: int,
                  device: torch.device | str, bsize: int) -> torch.Tensor:
    inp_sub = torch.tensor([0.5], device=device).view(1, -1, 1, 1)
    inp_div = torch.tensor([0.5], device=device).view(1, -1, 1, 1)
    gt_sub = torch.tensor([0.5], device=device).view(1, 1, -1)
    gt_div = torch.tensor([0.5], device=device).view(1, 1, -1)

    inp = (lr - inp_sub) / inp_div
    coord = utils.make_coord([target_h, target_w]).unsqueeze(0).to(device)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / target_h
    cell[:, :, 1] *= 2 / target_w

    with torch.no_grad():
        model.gen_feat(inp)
        preds = []
        for ql in range(0, coord.shape[1], bsize):
            qr = min(ql + bsize, coord.shape[1])
            preds.append(model.query_rgb(coord[:, ql:qr], cell[:, ql:qr]))
        pred = torch.cat(preds, dim=1)

    pred = (pred * gt_div + gt_sub).clamp(0, 1)
    return pred.view(1, target_h, target_w, 3).permute(0, 3, 1, 2).contiguous()


def bicubic_baseline(lr: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    return resize_tensor(lr, (target_h, target_w))


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    coeffs = img.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
    return (img * coeffs).sum(dim=1, keepdim=True)


def shave_valid(x: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0 or x.shape[-2] <= 2 * shave or x.shape[-1] <= 2 * shave:
        return x
    return x[..., shave:-shave, shave:-shave]


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def sobel_mag(y: torch.Tensor) -> torch.Tensor:
    kx = y.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    ky = y.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def local_variance(y: torch.Tensor, k: int = 7) -> torch.Tensor:
    pad = k // 2
    mean = F.avg_pool2d(y, k, stride=1, padding=pad)
    mean2 = F.avg_pool2d(y.square(), k, stride=1, padding=pad)
    return (mean2 - mean.square()).clamp_min(0)


def top_mask(x: torch.Tensor, frac: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, int(round(flat.numel() * (1 - frac))))
    thresh = torch.kthvalue(flat, k).values
    return x >= thresh


def bottom_mask(x: torch.Tensor, frac: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, int(round(flat.numel() * frac)))
    thresh = torch.kthvalue(flat, k).values
    return x <= thresh


def masked_rmse(diff: torch.Tensor, mask: torch.Tensor) -> float:
    vals = diff[mask.expand_as(diff)]
    if vals.numel() == 0:
        return float("nan")
    return float(torch.sqrt(vals.square().mean()).item())


def highpass_rmse(pred_y: torch.Tensor, gt_y: torch.Tensor) -> float:
    kernel = pred_y.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
    p = F.conv2d(pred_y, kernel, padding=1)
    g = F.conv2d(gt_y, kernel, padding=1)
    return float(torch.sqrt((p - g).square().mean()).item())


def fft_band_power(diff_y: torch.Tensor) -> Tuple[float, float, float]:
    arr = diff_y.squeeze().detach().float().cpu().numpy()
    if arr.ndim != 2:
        return float("nan"), float("nan"), float("nan")
    h, w = arr.shape
    spec = np.fft.fftshift(np.fft.fft2(arr))
    power = (np.abs(spec) ** 2) / max(1, (h * w) ** 2)
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx ** 2 + yy ** 2)
    bands = [
        radius < 0.15,
        (radius >= 0.15) & (radius < 0.35),
        radius >= 0.35,
    ]
    vals = []
    for mask in bands:
        vals.append(float(power[mask].mean()) if np.any(mask) else float("nan"))
    return vals[0], vals[1], vals[2]


def quality_metrics(pred: torch.Tensor, gt: torch.Tensor, scale: int) -> Dict[str, float]:
    pred_y = rgb_to_y(pred)
    gt_y = rgb_to_y(gt)
    pred_v = shave_valid(pred_y, scale)
    gt_v = shave_valid(gt_y, scale)
    diff = pred_v - gt_v
    mse = float(diff.square().mean().item())

    grad = sobel_mag(gt_v)
    var = local_variance(gt_v)
    edge = top_mask(grad, 0.20)
    texture = top_mask(var, 0.20)
    flat = bottom_mask(var, 0.20)
    fft_low, fft_mid, fft_high = fft_band_power(diff)

    try:
        pred_np = pred_v.detach().cpu().squeeze().numpy()
        gt_np = gt_v.detach().cpu().squeeze().numpy()
        win_size = min(11, pred_np.shape[0] if pred_np.shape[0] % 2 == 1 else pred_np.shape[0] - 1,
                       pred_np.shape[1] if pred_np.shape[1] % 2 == 1 else pred_np.shape[1] - 1)
        win_size = max(3, win_size)
        ssim_y = float(skimage_ssim(gt_np, pred_np, data_range=1.0, win_size=win_size))
    except Exception:
        ssim_y = float("nan")

    return {
        "psnr_y": psnr_from_mse(mse),
        "ssim_y": ssim_y,
        "rmse_y": math.sqrt(mse),
        "edge_rmse": masked_rmse(diff, edge),
        "texture_rmse": masked_rmse(diff, texture),
        "flat_rmse": masked_rmse(diff, flat),
        "highpass_rmse": highpass_rmse(pred_v, gt_v),
        "fft_low_power": fft_low,
        "fft_mid_power": fft_mid,
        "fft_high_power": fft_high,
    }


def consistency_metrics(direct: torch.Tensor, down_from_high: torch.Tensor, obs_scale: int) -> Dict[str, float]:
    direct_y = shave_valid(rgb_to_y(direct), obs_scale)
    down_y = shave_valid(rgb_to_y(down_from_high), obs_scale)
    diff = direct_y - down_y
    mse = float(diff.square().mean().item())
    grad = sobel_mag(down_y)
    var = local_variance(down_y)
    edge = top_mask(grad, 0.20)
    texture = top_mask(var, 0.20)
    flat = bottom_mask(var, 0.20)
    return {
        "consistency_psnr_y": psnr_from_mse(mse),
        "consistency_rmse_y": math.sqrt(mse),
        "edge_consistency_rmse": masked_rmse(diff, edge),
        "texture_consistency_rmse": masked_rmse(diff, texture),
        "flat_consistency_rmse": masked_rmse(diff, flat),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]], keys: Sequence[str],
              metric_names: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, object]] = []
    for group_key, items in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, group_key)}
        rec["n"] = len(items)
        for m in metric_names:
            vals = [float(r[m]) for r in items if r.get(m) not in (None, "") and not math.isnan(float(r[m]))]
            rec[m] = float(np.mean(vals)) if vals else float("nan")
        out.append(rec)
    return out


def run_quality(args, device, model_names: List[str], out_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    use_bicubic = args.include_bicubic
    active_names = (["Bicubic"] if use_bicubic else []) + model_names

    loaded = {name: load_model(name, device) for name in model_names}
    for dataset in parse_csv_list(args.datasets):
        images = list_images(dataset, args.max_images)
        for img_path in tqdm(images, desc=f"quality:{dataset}"):
            for scale in parse_scales(args.scales):
                lr, gt = make_lr_hr(img_path, scale, device)
                target_h, target_w = gt.shape[-2:]
                preds: Dict[str, torch.Tensor] = {}
                if use_bicubic:
                    preds["Bicubic"] = bicubic_baseline(lr, target_h, target_w)
                for model_name, model in loaded.items():
                    preds[model_name] = predict_image(model, lr, target_h, target_w, device, args.eval_bsize)
                for model_name in active_names:
                    metrics = quality_metrics(preds[model_name], gt, scale)
                    rows.append({
                        "model": model_name,
                        "dataset": dataset,
                        "image": img_path.name,
                        "scale": f"x{scale}",
                        **metrics,
                    })
                del lr, gt, preds
                torch.cuda.empty_cache()

    write_csv(out_dir / "quality_metrics.csv", rows)
    summary = summarize(
        rows,
        ["model", "dataset", "scale"],
        ["psnr_y", "ssim_y", "rmse_y", "edge_rmse", "texture_rmse", "flat_rmse",
         "highpass_rmse", "fft_low_power", "fft_mid_power", "fft_high_power"],
    )
    write_csv(out_dir / "quality_summary.csv", summary)
    return rows


def run_consistency(args, device, model_names: List[str], out_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    loaded = {name: load_model(name, device) for name in model_names}
    pairs = parse_pairs(args.consistency_pairs)

    for dataset in parse_csv_list(args.datasets):
        images = list_images(dataset, args.max_images)
        for img_path in tqdm(images, desc=f"consistency:{dataset}"):
            for source_scale, obs_scale in pairs:
                lr, hr = make_lr_hr(img_path, source_scale, device)
                h_hr, w_hr = hr.shape[-2:]
                h_obs = max(1, int(round(h_hr * obs_scale / source_scale)))
                w_obs = max(1, int(round(w_hr * obs_scale / source_scale)))
                for model_name, model in loaded.items():
                    high = predict_image(model, lr, h_hr, w_hr, device, args.eval_bsize)
                    direct = predict_image(model, lr, h_obs, w_obs, device, args.eval_bsize)
                    down = resize_tensor(high, (h_obs, w_obs))
                    metrics = consistency_metrics(direct, down, obs_scale)
                    rows.append({
                        "model": model_name,
                        "dataset": dataset,
                        "image": img_path.name,
                        "source_scale": f"x{source_scale}",
                        "observation_scale": f"x{obs_scale}",
                        **metrics,
                    })
                    del high, direct, down
                    torch.cuda.empty_cache()
                del lr, hr

    write_csv(out_dir / "consistency_metrics.csv", rows)
    summary = summarize(
        rows,
        ["model", "dataset", "source_scale", "observation_scale"],
        ["consistency_psnr_y", "consistency_rmse_y", "edge_consistency_rmse",
         "texture_consistency_rmse", "flat_consistency_rmse"],
    )
    write_csv(out_dir / "consistency_summary.csv", summary)
    return rows


def choose_texture_crop(gt: torch.Tensor, crop_size: int) -> Tuple[int, int]:
    _, _, h, w = gt.shape
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)
    score = local_variance(rgb_to_y(gt), k=9)
    pooled = F.avg_pool2d(score, kernel_size=(crop_h, crop_w), stride=max(1, crop_h // 4))
    idx = int(torch.argmax(pooled).item())
    _, _, ph, pw = pooled.shape
    y_idx = idx // pw
    x_idx = idx % pw
    stride = max(1, crop_h // 4)
    y0 = min(y_idx * stride, h - crop_h)
    x0 = min(x_idx * stride, w - crop_w)
    return int(y0), int(x0)


def crop_tensor(x: torch.Tensor, y0: int, x0: int, size: int) -> torch.Tensor:
    return x[..., y0:y0 + size, x0:x0 + size]


def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(arr, 0, 1)


def run_visuals(args, device, model_names: List[str], out_dir: Path) -> None:
    vis_dir = out_dir / "figures" / "visual_crops"
    ensure_dir(vis_dir)
    loaded = {name: load_model(name, device) for name in model_names}
    datasets = parse_csv_list(args.visual_datasets)
    scales = parse_scales(args.visual_scales)

    for dataset in datasets:
        requested = parse_csv_list(args.visual_images)
        if requested:
            image_paths = [DATASETS[dataset] / name for name in requested if (DATASETS[dataset] / name).exists()]
        else:
            image_paths = list_images(dataset, args.visual_max_images)
        for img_path in image_paths:
            for scale in scales:
                lr, gt = make_lr_hr(img_path, scale, device)
                h, w = gt.shape[-2:]
                preds: Dict[str, torch.Tensor] = {
                    "GT": gt,
                    "Bicubic": bicubic_baseline(lr, h, w),
                }
                for model_name, model in loaded.items():
                    preds[model_name] = predict_image(model, lr, h, w, device, args.eval_bsize)

                y0, x0 = choose_texture_crop(gt, args.crop_size)
                crop_size = min(args.crop_size, h - y0, w - x0)
                names = ["GT", "Bicubic"] + model_names
                fig, axes = plt.subplots(2, len(names), figsize=(2.2 * len(names), 4.6))
                for col, name in enumerate(names):
                    crop = crop_tensor(preds[name], y0, x0, crop_size)
                    axes[0, col].imshow(to_numpy_img(crop))
                    axes[0, col].set_title(name, fontsize=9)
                    axes[0, col].axis("off")
                    if name == "GT":
                        axes[1, col].imshow(np.zeros((crop_size, crop_size)), cmap="magma", vmin=0, vmax=0.12)
                    else:
                        err = torch.abs(rgb_to_y(crop) - rgb_to_y(crop_tensor(gt, y0, x0, crop_size)))
                        axes[1, col].imshow(err.detach().cpu().squeeze().numpy(), cmap="magma", vmin=0, vmax=0.12)
                    axes[1, col].axis("off")
                fig.suptitle(f"{dataset}/{img_path.name} x{scale} crop=({y0},{x0})", fontsize=10)
                plt.tight_layout()
                out_path = vis_dir / f"{dataset}_{img_path.stem}_x{scale}_crop.png"
                fig.savefig(out_path, dpi=180)
                plt.close(fig)
                del lr, gt, preds
                torch.cuda.empty_cache()


def plot_summary(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    quality_path = out_dir / "quality_summary.csv"
    if quality_path.exists():
        rows = list(csv.DictReader(quality_path.open()))
        for metric, ylabel, filename in [
            ("psnr_y", "PSNR-Y (dB)", "quality_psnr_y.png"),
            ("texture_rmse", "Texture RMSE-Y", "texture_rmse_y.png"),
            ("highpass_rmse", "High-pass RMSE-Y", "highpass_rmse_y.png"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            labels = sorted(set((r["dataset"], r["scale"]) for r in rows))
            models_seen = [m for m in STYLE if any(r["model"] == m for r in rows)]
            x = np.arange(len(labels))
            width = 0.8 / max(1, len(models_seen))
            for i, model_name in enumerate(models_seen):
                vals = []
                for dataset, scale in labels:
                    match = [r for r in rows if r["model"] == model_name and r["dataset"] == dataset and r["scale"] == scale]
                    vals.append(float(match[0][metric]) if match else np.nan)
                ax.bar(x + (i - len(models_seen) / 2) * width + width / 2, vals, width,
                       label=model_name, color=STYLE.get(model_name, "#888888"))
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d}\n{s}" for d, s in labels], fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8, ncol=3)
            fig.tight_layout()
            fig.savefig(fig_dir / filename, dpi=180)
            plt.close(fig)

    consistency_path = out_dir / "consistency_summary.csv"
    if consistency_path.exists():
        rows = list(csv.DictReader(consistency_path.open()))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = sorted(set((r["dataset"], r["source_scale"], r["observation_scale"]) for r in rows))
        models_seen = [m for m in STYLE if any(r["model"] == m for r in rows) and m != "Bicubic"]
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(models_seen))
        for i, model_name in enumerate(models_seen):
            vals = []
            for dataset, src, obs in labels:
                match = [
                    r for r in rows
                    if r["model"] == model_name and r["dataset"] == dataset
                    and r["source_scale"] == src and r["observation_scale"] == obs
                ]
                vals.append(float(match[0]["consistency_psnr_y"]) if match else np.nan)
            ax.bar(x + (i - len(models_seen) / 2) * width + width / 2, vals, width,
                   label=model_name, color=STYLE.get(model_name, "#888888"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}\n{src}->{obs}" for d, src, obs in labels], fontsize=8)
        ax.set_ylabel("Cross-scale consistency PSNR-Y (dB)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(fig_dir / "consistency_psnr_y.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="LIIF,LTE,LTE-NoCellPhase,LTE-PhaseZ,SC-INR-FixedOmega,SC-INR-NoPhi,SC-INR")
    parser.add_argument("--datasets", default="bsd100,urban100")
    parser.add_argument("--scales", default="4,8,16,30")
    parser.add_argument("--consistency_pairs", default="8,4;16,4;30,4")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_bsize", type=int, default=50000)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "analysis" / "seed1_aux_metrics")
    parser.add_argument("--include_bicubic", action="store_true", default=True)
    parser.add_argument("--skip_quality", action="store_true")
    parser.add_argument("--skip_consistency", action="store_true")
    parser.add_argument("--skip_visuals", action="store_true")
    parser.add_argument("--visual_datasets", default="set14,urban100")
    parser.add_argument("--visual_images", default="barbara.png,comic.png,img_004.png")
    parser.add_argument("--visual_scales", default="8,16")
    parser.add_argument("--visual_max_images", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=96)
    args = parser.parse_args()

    os.chdir(ROOT)
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model_names = parse_csv_list(args.models)
    missing = [m for m in model_names if m not in MODEL_PATHS or not MODEL_PATHS[m].exists()]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints: {missing}")

    meta = {
        "models": model_names,
        "datasets": parse_csv_list(args.datasets),
        "scales": parse_scales(args.scales),
        "consistency_pairs": parse_pairs(args.consistency_pairs),
        "max_images": args.max_images,
        "device": str(device),
        "eval_bsize": args.eval_bsize,
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2))

    if not args.skip_quality:
        run_quality(args, device, model_names, out_dir)
    if not args.skip_consistency:
        run_consistency(args, device, model_names, out_dir)
    if not args.skip_visuals:
        run_visuals(args, device, model_names, out_dir)
    plot_summary(out_dir)
    print(f"Auxiliary metrics written to {out_dir}")


if __name__ == "__main__":
    main()
