#!/usr/bin/env python3
"""Create paper-style qualitative ASISR crop figures.

This script is intended for SR paper figures, not debugging dashboards. It
uses the same HR-on-the-fly downsampling protocol as the benchmark, draws a
crop rectangle on the full HR image, renders selected method crops, and writes
local crop metrics. Use manual crop coordinates for final figures; automatic
texture crop is only a starting point for finding candidates.

Example:
    python scripts/viz/prepare_qualitative_figure.py \
        --dataset urban100 --image img_004.png --scale 8 \
        --models Bicubic,LTE,SC-INR-NoPhi,SC-INR \
        --auto_texture_crop --out artifacts/derived/paper_figures/qualitative_selected_seed1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT))

from evaluate_seed1_aux_metrics import (
    DATASETS,
    MODEL_PATHS,
    bicubic_baseline,
    choose_texture_crop,
    crop_tensor,
    ensure_dir,
    load_model,
    make_lr_hr,
    predict_image,
    psnr_from_mse,
    rgb_to_y,
    shave_valid,
    to_numpy_img,
)


def parse_models(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_crop(text: str | None) -> Tuple[int, int, int] | None:
    if not text:
        return None
    parts = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--crop must be formatted as y,x,size")
    return parts[0], parts[1], parts[2]


def y_psnr(pred: torch.Tensor, gt: torch.Tensor, scale: int) -> float:
    pred_y = shave_valid(rgb_to_y(pred), scale)
    gt_y = shave_valid(rgb_to_y(gt), scale)
    mse = float((pred_y - gt_y).square().mean().item())
    return psnr_from_mse(mse)


def crop_metrics(preds: Dict[str, torch.Tensor], gt: torch.Tensor, scale: int,
                 y0: int, x0: int, crop_size: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    gt_crop = crop_tensor(gt, y0, x0, crop_size)
    for name, pred in preds.items():
        if name == "GT":
            continue
        pred_crop = crop_tensor(pred, y0, x0, crop_size)
        diff = rgb_to_y(pred_crop) - rgb_to_y(gt_crop)
        rmse = float(torch.sqrt(diff.square().mean()).item())
        rows.append({
            "model": name,
            "crop_psnr_y": f"{y_psnr(pred_crop, gt_crop, scale):.4f}",
            "crop_rmse_y": f"{rmse:.6f}",
        })
    return rows


def crop_psnr_at(x: torch.Tensor, gt: torch.Tensor, scale: int,
                 y0: int, x0: int, crop_size: int) -> float:
    return y_psnr(crop_tensor(x, y0, x0, crop_size), crop_tensor(gt, y0, x0, crop_size), scale)


def texture_score(gt: torch.Tensor, y0: int, x0: int, crop_size: int) -> float:
    crop_y = rgb_to_y(crop_tensor(gt, y0, x0, crop_size))
    return float(crop_y.var(unbiased=False).item())


def find_delta_crop(args, gt: torch.Tensor, preds: Dict[str, torch.Tensor]) -> Tuple[int, int, int, List[Dict[str, object]]]:
    if args.target_model not in preds:
        raise ValueError(f"--target_model {args.target_model!r} is not in predictions")
    if args.baseline_model not in preds:
        raise ValueError(f"--baseline_model {args.baseline_model!r} is not in predictions")

    _, _, h, w = gt.shape
    size = min(args.crop_size, h, w)
    stride = args.crop_stride if args.crop_stride > 0 else max(1, size // 2)
    ys = list(range(0, max(1, h - size + 1), stride))
    xs = list(range(0, max(1, w - size + 1), stride))
    if ys[-1] != h - size:
        ys.append(h - size)
    if xs[-1] != w - size:
        xs.append(w - size)

    rows: List[Dict[str, object]] = []
    for y0 in ys:
        for x0 in xs:
            target_psnr = crop_psnr_at(preds[args.target_model], gt, args.scale, y0, x0, size)
            baseline_psnr = crop_psnr_at(preds[args.baseline_model], gt, args.scale, y0, x0, size)
            sc_psnr = crop_psnr_at(preds["SC-INR"], gt, args.scale, y0, x0, size) if "SC-INR" in preds else float("nan")
            tex = texture_score(gt, y0, x0, size)
            rows.append({
                "y": y0,
                "x": x0,
                "size": size,
                "target_model": args.target_model,
                "baseline_model": args.baseline_model,
                "target_psnr_y": target_psnr,
                "baseline_psnr_y": baseline_psnr,
                "sc_inr_psnr_y": sc_psnr,
                "delta_target_vs_baseline": target_psnr - baseline_psnr,
                "delta_target_vs_sc_inr": target_psnr - sc_psnr if not math.isnan(sc_psnr) else float("nan"),
                "texture_var_y": tex,
            })

    if not rows:
        raise RuntimeError("no crop candidates found")
    tex_values = torch.tensor([float(r["texture_var_y"]) for r in rows])
    if args.min_texture_quantile > 0:
        q = min(max(args.min_texture_quantile, 0.0), 1.0)
        thresh = float(torch.quantile(tex_values, q).item())
        filtered = [r for r in rows if float(r["texture_var_y"]) >= thresh]
    else:
        filtered = rows
    if not filtered:
        filtered = rows

    sort_key = lambda r: (float(r["delta_target_vs_baseline"]), float(r["texture_var_y"]))
    filtered.sort(key=sort_key, reverse=True)
    rows.sort(key=sort_key, reverse=True)
    best = filtered[0]
    return int(best["y"]), int(best["x"]), int(best["size"]), rows


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_predictions(args, device: torch.device, methods: List[str]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    img_path = DATASETS[args.dataset] / args.image
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")

    lr, gt = make_lr_hr(img_path, args.scale, device)
    h, w = gt.shape[-2:]
    preds: Dict[str, torch.Tensor] = {"GT": gt}
    if "Bicubic" in methods:
        preds["Bicubic"] = bicubic_baseline(lr, h, w)

    model_names = [m for m in methods if m not in {"GT", "Bicubic"}]
    missing = [m for m in model_names if m not in MODEL_PATHS or not MODEL_PATHS[m].exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints for models: {missing}")

    for name in model_names:
        model = load_model(name, device)
        preds[name] = predict_image(model, lr, h, w, device, args.eval_bsize)
        del model
        torch.cuda.empty_cache()
    return lr, gt, preds


def resolve_crop(args, gt: torch.Tensor) -> Tuple[int, int, int]:
    _, _, h, w = gt.shape
    parsed = parse_crop(args.crop)
    if parsed is None:
        if not args.auto_texture_crop:
            raise ValueError("provide --crop y,x,size or use --auto_texture_crop")
        y0, x0 = choose_texture_crop(gt, args.crop_size)
        size = args.crop_size
    else:
        y0, x0, size = parsed

    size = min(size, h, w)
    y0 = max(0, min(y0, h - size))
    x0 = max(0, min(x0, w - size))
    return int(y0), int(x0), int(size)


def draw_figure(args, gt: torch.Tensor, preds: Dict[str, torch.Tensor],
                methods: List[str], y0: int, x0: int, crop_size: int,
                out_base: Path) -> None:
    names = ["GT"] + [m for m in methods if m != "GT"]
    names = list(dict.fromkeys(names))

    n_crops = len(names)
    fig_w = max(10.0, 3.2 + 1.55 * n_crops)
    fig_h = 3.4 if not args.with_error_maps else 5.2
    rows = 1 if not args.with_error_maps else 2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(rows, n_crops + 2, width_ratios=[2.2, 0.08] + [1] * n_crops)

    full_ax = fig.add_subplot(gs[:, 0])
    full_ax.imshow(to_numpy_img(gt))
    full_ax.add_patch(Rectangle((x0, y0), crop_size, crop_size, fill=False,
                                edgecolor="#d62728", linewidth=2.2))
    full_ax.set_title(f"{args.dataset}/{args.image} x{args.scale}", fontsize=9)
    full_ax.axis("off")

    gt_crop = crop_tensor(gt, y0, x0, crop_size)
    vmax = args.error_vmax
    metric_lookup = {r["model"]: r for r in crop_metrics(preds, gt, args.scale, y0, x0, crop_size)}
    for col, name in enumerate(names):
        crop = crop_tensor(preds[name], y0, x0, crop_size)
        ax = fig.add_subplot(gs[0, col + 2])
        ax.imshow(to_numpy_img(crop))
        title = name
        if args.show_crop_psnr and name != "GT":
            title += f"\n{metric_lookup[name]['crop_psnr_y']} dB"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

        if args.with_error_maps:
            err_ax = fig.add_subplot(gs[1, col + 2])
            if name == "GT":
                err = np.zeros((crop_size, crop_size), dtype=np.float32)
            else:
                err_t = torch.abs(rgb_to_y(crop) - rgb_to_y(gt_crop))
                err = err_t.detach().cpu().squeeze().numpy()
            err_ax.imshow(err, cmap="magma", vmin=0, vmax=vmax)
            err_ax.axis("off")

    fig.subplots_adjust(left=0.015, right=0.995, top=0.88, bottom=0.035, wspace=0.12, hspace=0.12)
    fig.savefig(out_base.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def save_individual_crops(out_dir: Path, preds: Dict[str, torch.Tensor],
                          y0: int, x0: int, crop_size: int) -> None:
    from PIL import Image

    crop_dir = out_dir / "crops"
    ensure_dir(crop_dir)
    for name, pred in preds.items():
        arr = (to_numpy_img(crop_tensor(pred, y0, x0, crop_size)) * 255.0).round().astype(np.uint8)
        Image.fromarray(arr).save(crop_dir / f"{name.replace('+', 'plus').replace('/', '_')}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-style qualitative crop figures for ASISR checkpoints."
    )
    parser.add_argument("--dataset", default="urban100", choices=sorted(DATASETS.keys()))
    parser.add_argument("--image", default="img_004.png")
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--models", default="Bicubic,LTE,SC-INR-NoPhi,SC-INR")
    parser.add_argument("--crop", default=None, help="Manual crop as y,x,size in HR/output coordinates.")
    parser.add_argument("--auto_texture_crop", action="store_true", help="Use highest local-variance crop as a candidate.")
    parser.add_argument("--auto_delta_crop", action="store_true",
                        help="Select a crop by local target-vs-baseline PSNR gain; writes ranked candidates.")
    parser.add_argument("--target_model", default="SC-INR")
    parser.add_argument("--baseline_model", default="LTE")
    parser.add_argument("--crop_stride", type=int, default=0, help="Grid stride for --auto_delta_crop. Default: crop_size/2.")
    parser.add_argument("--min_texture_quantile", type=float, default=0.50,
                        help="Keep only crops above this texture variance quantile for --auto_delta_crop.")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_bsize", type=int, default=50000)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "derived" / "paper_figures" / "qualitative_selected_seed1")
    parser.add_argument("--name", default=None, help="Output basename. Defaults to dataset_image_xscale_crop.")
    parser.add_argument("--with_error_maps", action="store_true")
    parser.add_argument("--show_crop_psnr", action="store_true")
    parser.add_argument("--error_vmax", type=float, default=0.12)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--max_candidate_rows", type=int, default=200)
    args = parser.parse_args()

    os.chdir(ROOT)
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    methods = parse_models(args.models)
    _, gt, preds = make_predictions(args, device, methods)
    candidate_rows: List[Dict[str, object]] = []
    if args.auto_delta_crop:
        y0, x0, crop_size, candidate_rows = find_delta_crop(args, gt, preds)
    else:
        y0, x0, crop_size = resolve_crop(args, gt)

    stem = args.name or f"{args.dataset}_{Path(args.image).stem}_x{args.scale}_crop_y{y0}_x{x0}_s{crop_size}"
    out_base = out_dir / stem
    draw_figure(args, gt, preds, methods, y0, x0, crop_size, out_base)
    save_individual_crops(out_dir / stem, preds, y0, x0, crop_size)

    metric_rows = crop_metrics(preds, gt, args.scale, y0, x0, crop_size)
    save_csv(out_base.with_name(out_base.name + "_metrics.csv"), metric_rows)
    if candidate_rows:
        save_csv(out_base.with_name(out_base.name + "_crop_candidates.csv"), [
            {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            for row in candidate_rows[:args.max_candidate_rows]
        ])
    meta = {
        "dataset": args.dataset,
        "image": args.image,
        "scale": args.scale,
        "models": methods,
        "crop_y": y0,
        "crop_x": x0,
        "crop_size": crop_size,
        "auto_delta_crop": args.auto_delta_crop,
        "target_model": args.target_model,
        "baseline_model": args.baseline_model,
        "device": str(device),
        "output_png": str(out_base.with_suffix(".png")),
        "output_pdf": str(out_base.with_suffix(".pdf")),
    }
    out_base.with_name(out_base.name + "_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
