#!/usr/bin/env python3
"""Sampling-response diagnostics for SC-INR-style decoders.

This script is meant for mechanism analysis, not for benchmark reporting. It
produces compact CSV/figures for two questions:

1. What omega and sinc-response distributions does a model produce at ID/OOD
   output scales?
2. How much does a model's RGB output change when only the output cell changes
   for the same LR input and fixed query coordinates?

Outputs are written under results/analysis/sampling_response/ by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
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
    "LTE-NoCell": ROOT / "save" / "lte-no-cell" / "epoch-best.pth",
    "LTE-FeaturePhase": ROOT / "save" / "lte-feature-phase" / "epoch-best.pth",
    "SC-INR-Fixed": ROOT / "save" / "sc-inr-fixed" / "epoch-best.pth",
    "SC-INR-Adaptive": ROOT / "save" / "sc-inr-adaptive" / "epoch-best.pth",
    "SC-INR-Adaptive-Signed": ROOT / "save" / "sc-inr-adaptive-signed" / "epoch-best.pth",
}

STYLE = {
    "LIIF": "#4C72B0",
    "LTE": "#DD8452",
    "LTE-NoCell": "#55A868",
    "LTE-FeaturePhase": "#8172B2",
    "SC-INR-Fixed": "#C44E52",
    "SC-INR-Adaptive": "#8B0000",
    "SC-INR-Adaptive-Signed": "#222222",
}


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_scales(text: str) -> List[int]:
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


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


def make_lr_hr(img_path: Path, scale: float, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    img_hr_pil = Image.open(img_path).convert("RGB")
    w_hr, h_hr = img_hr_pil.size
    h_lr = max(1, int(math.floor(h_hr / scale + 1e-9)))
    w_lr = max(1, int(math.floor(w_hr / scale + 1e-9)))
    target_h = int(round(h_lr * scale))
    target_w = int(round(w_lr * scale))
    hr_crop = img_hr_pil.crop((0, 0, target_w, target_h))
    lr = hr_crop.resize((w_lr, h_lr), Image.BICUBIC)
    return pil_to_tensor(lr, device), pil_to_tensor(hr_crop, device)


def rgb_to_y(pred: torch.Tensor) -> torch.Tensor:
    coeffs = pred.new_tensor([65.738, 129.057, 25.064]).view(1, 1, 3) / 256
    return (pred * coeffs).sum(dim=-1, keepdim=True)


def load_model(model_name: str, device: torch.device | str):
    path = MODEL_PATHS[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_name}: {path}")
    ckpt = torch.load(path, map_location="cpu")
    model = models.make(ckpt["model"], load_sd=True, strict=False).to(device)
    model.eval()
    return model


def summarize_tensor(x: torch.Tensor, prefix: str) -> Dict[str, float]:
    vals = x.detach().float().flatten().cpu()
    if vals.numel() == 0:
        return {f"{prefix}_{k}": float("nan") for k in ["mean", "std", "min", "q01", "q05", "q50", "q95", "q99", "max"]}
    qs = torch.quantile(vals, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]))
    return {
        f"{prefix}_mean": float(vals.mean()),
        f"{prefix}_std": float(vals.std(unbiased=False)),
        f"{prefix}_min": float(vals.min()),
        f"{prefix}_q01": float(qs[0]),
        f"{prefix}_q05": float(qs[1]),
        f"{prefix}_q50": float(qs[2]),
        f"{prefix}_q95": float(qs[3]),
        f"{prefix}_q99": float(qs[4]),
        f"{prefix}_max": float(vals.max()),
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


def omega_components(model) -> torch.Tensor | None:
    if hasattr(model, "omega_map"):
        raw = model.omega_map.detach()
        bs, ch, h, w = raw.shape
        if not hasattr(model, "num_freqs"):
            return None
        return raw.view(bs, model.num_freqs, 2, h, w).permute(0, 3, 4, 1, 2).contiguous()
    if hasattr(model, "freqs"):
        freqs = model.freqs.detach().view(1, 1, 1, model.num_freqs, 2)
        return freqs
    return None


def has_sc_inr_omega(model) -> bool:
    return hasattr(model, "freqs") or (hasattr(model, "omega_conv") and hasattr(model, "num_freqs"))


def run_response_distribution(args, device: torch.device, out_dir: Path) -> None:
    rows: List[Dict[str, object]] = []
    model_names = parse_csv_list(args.models)
    scales = parse_scales(args.scales)

    for model_name in model_names:
        model = load_model(model_name, device)
        if not has_sc_inr_omega(model):
            print(f"[SKIP] {model_name}: no SC-INR omega representation found")
            continue
        for dataset in parse_csv_list(args.datasets):
            for img_path in list_images(dataset, args.max_images):
                lr, _ = make_lr_hr(img_path, args.lr_scale, device)
                with torch.no_grad():
                    model.gen_feat((lr - 0.5) / 0.5)
                    omega = omega_components(model)
                    if omega is None:
                        continue
                    omega_x = omega[..., 0]
                    omega_y = omega[..., 1]
                    omega_mag = torch.sqrt(omega_x.square() + omega_y.square())
                    omega_abs = torch.cat([omega_x.abs().flatten(), omega_y.abs().flatten()])
                    neg_frac = float(((omega_x < 0).float().mean() + (omega_y < 0).float().mean()) / 2)
                    bound = getattr(model, "omega_bound", None)
                    if bound is None:
                        near_bound_frac = float("nan")
                    else:
                        near_bound_frac = float((omega_abs > 0.95 * float(bound)).float().mean())

                    for scale in scales:
                        rel_cell = 2.0 / float(scale)
                        sinc_x = torch.sinc(omega_x * rel_cell / 2)
                        sinc_y = torch.sinc(omega_y * rel_cell / 2)
                        response = sinc_x * sinc_y
                        row: Dict[str, object] = {
                            "model": model_name,
                            "dataset": dataset,
                            "image": img_path.name,
                            "lr_scale": f"x{args.lr_scale}",
                            "observation_scale": f"x{scale}",
                            "omega_param": getattr(model, "omega_param", "fixed"),
                            "omega_bound": float(bound) if bound is not None else "",
                            "omega_neg_frac": neg_frac,
                            "omega_near_bound_frac": near_bound_frac,
                            "response_neg_frac": float((response < 0).float().mean()),
                        }
                        row.update(summarize_tensor(omega_x, "omega_x"))
                        row.update(summarize_tensor(omega_y, "omega_y"))
                        row.update(summarize_tensor(omega_mag, "omega_mag"))
                        row.update(summarize_tensor(response, "response"))
                        rows.append(row)
                del lr
                torch.cuda.empty_cache()

    write_csv(out_dir / "response_distribution.csv", rows)
    plot_response_distribution(out_dir / "response_distribution.csv", out_dir / "figures")


def plot_response_distribution(csv_path: Path, fig_dir: Path) -> None:
    if not csv_path.exists():
        return
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    ensure_dir(fig_dir)
    grouped = (
        df.groupby(["model", "observation_scale"], as_index=False)
        [["response_mean", "response_q05", "response_q50", "response_q95", "response_neg_frac", "omega_neg_frac"]]
        .mean()
    )
    scale_nums = grouped["observation_scale"].str.replace("x", "", regex=False).astype(float)
    grouped = grouped.assign(scale_num=scale_nums).sort_values(["model", "scale_num"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model_name, sub in grouped.groupby("model"):
        ax.plot(
            sub["scale_num"],
            sub["response_mean"],
            marker="o",
            color=STYLE.get(model_name, "#888888"),
            label=model_name,
        )
        ax.fill_between(
            sub["scale_num"].to_numpy(),
            sub["response_q05"].to_numpy(),
            sub["response_q95"].to_numpy(),
            color=STYLE.get(model_name, "#888888"),
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlabel("Observation scale")
    ax.set_ylabel("Sinc response mean with 5-95% band")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "response_mean_by_scale.png", dpi=220)
    fig.savefig(fig_dir / "response_mean_by_scale.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model_name, sub in grouped.groupby("model"):
        ax.plot(
            sub["scale_num"],
            sub["response_neg_frac"],
            marker="o",
            color=STYLE.get(model_name, "#888888"),
            label=model_name,
        )
    ax.set_xlabel("Observation scale")
    ax.set_ylabel("Fraction of negative sinc responses")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "response_negative_fraction.png", dpi=220)
    fig.savefig(fig_dir / "response_negative_fraction.pdf")
    plt.close(fig)


def make_query_coords(size: Sequence[int], max_queries: int, device: torch.device) -> torch.Tensor:
    coord = utils.make_coord(size).unsqueeze(0).to(device)
    if coord.shape[1] > max_queries:
        idx = torch.linspace(0, coord.shape[1] - 1, steps=max_queries, device=device).long()
        coord = coord[:, idx, :]
    return coord.contiguous()


def query_model(model, lr: torch.Tensor, coord: torch.Tensor, scale: int, bsize: int) -> torch.Tensor:
    h_lr, w_lr = lr.shape[-2:]
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / int(round(h_lr * scale))
    cell[:, :, 1] *= 2 / int(round(w_lr * scale))
    with torch.no_grad():
        model.gen_feat((lr - 0.5) / 0.5)
        preds = []
        for ql in range(0, coord.shape[1], bsize):
            qr = min(ql + bsize, coord.shape[1])
            pred = model.query_rgb(coord[:, ql:qr], cell[:, ql:qr])
            preds.append(pred)
        pred = torch.cat(preds, dim=1)
    return (pred * 0.5 + 0.5).clamp(0, 1)


def run_cell_curve(args, device: torch.device, out_dir: Path) -> None:
    rows: List[Dict[str, object]] = []
    model_names = parse_csv_list(args.cell_models)
    scales = parse_scales(args.cell_scales)

    image_path = DATASETS[args.cell_dataset] / args.cell_image if args.cell_image else list_images(args.cell_dataset, 1)[0]
    lr, hr = make_lr_hr(image_path, args.cell_lr_scale, device)
    coord = make_query_coords(hr.shape[-2:], args.max_queries, device)

    for model_name in model_names:
        model = load_model(model_name, device)
        preds: Dict[int, torch.Tensor] = {}
        for scale in scales:
            preds[scale] = query_model(model, lr, coord, scale, args.eval_bsize)
        ref = preds[args.cell_ref_scale if args.cell_ref_scale in preds else scales[0]]
        prev = None
        for scale in scales:
            pred = preds[scale]
            pred_y = rgb_to_y(pred)
            ref_y = rgb_to_y(ref)
            delta_ref = torch.sqrt((pred_y - ref_y).square().mean())
            if prev is None:
                delta_adj = torch.tensor(float("nan"), device=device)
            else:
                delta_adj = torch.sqrt((pred_y - rgb_to_y(prev)).square().mean())
            rows.append({
                "model": model_name,
                "dataset": args.cell_dataset,
                "image": image_path.name,
                "lr_scale": f"x{args.cell_lr_scale}",
                "observation_scale": f"x{scale}",
                "ref_scale": f"x{args.cell_ref_scale if args.cell_ref_scale in preds else scales[0]}",
                "mean_y": float(pred_y.mean()),
                "std_y": float(pred_y.std(unbiased=False)),
                "rmse_y_vs_ref_cell": float(delta_ref),
                "rmse_y_vs_previous_cell": float(delta_adj),
            })
            prev = pred
        del model
        torch.cuda.empty_cache()

    write_csv(out_dir / "cell_response_curve.csv", rows)
    plot_cell_curve(out_dir / "cell_response_curve.csv", out_dir / "figures")


def plot_cell_curve(csv_path: Path, fig_dir: Path) -> None:
    if not csv_path.exists():
        return
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    ensure_dir(fig_dir)
    df["scale_num"] = df["observation_scale"].str.replace("x", "", regex=False).astype(float)
    df = df.sort_values(["model", "scale_num"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model_name, sub in df.groupby("model"):
        ax.plot(
            sub["scale_num"],
            sub["rmse_y_vs_ref_cell"],
            marker="o",
            color=STYLE.get(model_name, "#888888"),
            label=model_name,
        )
    ax.set_xlabel("Observation scale used only for cell")
    ax.set_ylabel("Output RMSE-Y vs. reference cell")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "cell_response_delta_vs_ref.png", dpi=220)
    fig.savefig(fig_dir / "cell_response_delta_vs_ref.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["response", "cell", "both"], default="both")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "analysis" / "sampling_response")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="SC-INR-Fixed,SC-INR-Adaptive")
    parser.add_argument("--datasets", default="bsd100,urban100")
    parser.add_argument("--max_images", type=int, default=5)
    parser.add_argument("--lr_scale", type=int, default=4)
    parser.add_argument("--scales", default="4,8,16,30")
    parser.add_argument("--cell_models", default="LTE,LTE-NoCell,LTE-FeaturePhase,SC-INR-Fixed,SC-INR-Adaptive")
    parser.add_argument("--cell_dataset", default="urban100")
    parser.add_argument("--cell_image", default="img_004.png")
    parser.add_argument("--cell_lr_scale", type=int, default=4)
    parser.add_argument("--cell_ref_scale", type=int, default=4)
    parser.add_argument("--cell_scales", default="4,6,8,12,16,24,30")
    parser.add_argument("--max_queries", type=int, default=1024)
    parser.add_argument("--eval_bsize", type=int, default=50000)
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.mode in {"response", "both"}:
        run_response_distribution(args, device, out_dir)
    if args.mode in {"cell", "both"}:
        run_cell_curve(args, device, out_dir)
    print(f"Sampling-response diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
