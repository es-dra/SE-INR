#!/usr/bin/env python3
"""评估 SC-INR 的逐图统计和预注册局部优势区间。

本脚本服务论文证据链补强，不用于挑选单张好图后倒推结论。它做两件事：

1. 逐图逐尺度 paired statistics：检查 3-seed 平均增益是否由少数图支撑。
2. seed1 局部 crop 候选池：用只依赖 GT 的 texture/edge/highpass 分层，观察
   SC-INR 在哪些局部区域更常优于 LTE/LIIF/NoPhi/NoSinc。

所有 crop 坐标和分层只由 GT 决定；正文可选图必须来自完整候选池，并保留
top/neutral/failure 代表，避免只展示最有利区域。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from evaluate_seed1_aux_metrics import (  # noqa: E402
    DATASETS,
    bicubic_baseline,
    bottom_mask,
    crop_tensor,
    fft_band_power,
    highpass_rmse,
    list_images,
    load_model,
    local_variance,
    make_lr_hr,
    masked_rmse,
    predict_image,
    psnr_from_mse,
    quality_metrics,
    rgb_to_y,
    shave_valid,
    sobel_mag,
    to_numpy_img,
    top_mask,
)
from scripts.analysis.model_registry import STYLE  # noqa: E402


MAIN_MODELS = ["LIIF", "LTE", "SC-INR"]
LOCAL_MODELS = ["LIIF", "LTE", "SC-INR", "SC-INR-NoPhi", "SC-INR-NoSinc"]
LOCAL_METHODS = ["Bicubic"] + LOCAL_MODELS
PAPER_EXTERNAL_MODELS = ["LIIF", "LTE", "SC-INR"]
DATASET_DISPLAY = {
    "set5": "Set5",
    "set14": "Set14",
    "bsd100": "BSD100",
    "urban100": "Urban100",
}
SPLIT_BY_SCALE = {2: "ID", 3: "ID", 4: "ID", 6: "OOD", 8: "OOD", 12: "OOD", 16: "OOD", 24: "OOD", 30: "OOD"}


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> List[int]:
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def quantile(values: Sequence[float], q: float) -> float:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def median(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def std(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else float("nan")


def sem(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(std(vals) / math.sqrt(len(vals))) if vals else float("nan")


def model_path(model_name: str, seed: int) -> Path:
    model_dirs = {
        "LIIF": "liif",
        "LTE": "lte",
        "SC-INR": "sc-inr",
        "SC-INR-NoPhi": "sc-inr-nophi",
        "SC-INR-NoSinc": "sc-inr-nosinc",
    }
    if model_name not in model_dirs:
        raise KeyError(f"unsupported model for seed path: {model_name}")
    return ROOT / "artifacts" / "checkpoints" / f"seed{seed}" / model_dirs[model_name] / "epoch-best.pth"


def load_model_from_seed(model_name: str, seed: int, device: torch.device):
    if seed == 1:
        return load_model(model_name, device)
    path = model_path(model_name, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    import models  # noqa: PLC0415

    model = models.make(ckpt["model"], load_sd=True, strict=False).to(device)
    model.eval()
    return model


def split_for_scale(scale: int) -> str:
    return SPLIT_BY_SCALE.get(scale, "OOD" if scale > 4 else "ID")


def image_limit(paths: List[Path], max_images: int) -> List[Path]:
    if max_images and max_images > 0:
        return paths[:max_images]
    return paths


def quality_row(model_name: str, dataset: str, image: str, scale: int, metrics: Dict[str, float], seed: int) -> Dict[str, object]:
    return {
        "seed": seed,
        "dataset": DATASET_DISPLAY.get(dataset, dataset),
        "image": image,
        "scale": f"x{scale}",
        "scale_num": scale,
        "split": split_for_scale(scale),
        "model": model_name,
        **{k: f"{v:.8f}" if isinstance(v, float) and math.isfinite(v) else v for k, v in metrics.items()},
    }


def run_per_image(args, device: torch.device, out_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    datasets = parse_csv_list(args.per_image_datasets)
    scales = parse_ints(args.per_image_scales)
    seeds = parse_ints(args.seeds)
    models = parse_csv_list(args.per_image_models)

    for seed in seeds:
        loaded = {name: load_model_from_seed(name, seed, device) for name in models}
        for dataset in datasets:
            images = image_limit(list_images(dataset, args.per_image_max_images), args.per_image_max_images)
            for img_path in tqdm(images, desc=f"per-image seed{seed}:{dataset}"):
                for scale in scales:
                    lr, gt = make_lr_hr(img_path, scale, device)
                    h, w = gt.shape[-2:]
                    for model_name, model in loaded.items():
                        pred = predict_image(model, lr, h, w, device, args.eval_bsize)
                        metrics = quality_metrics(pred, gt, scale)
                        rows.append(quality_row(model_name, dataset, img_path.name, scale, metrics, seed))
                        del pred
                    del lr, gt
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_dir / "per_image_per_scale_metrics.csv", rows)
    return rows


def paired_delta_rows(rows: List[Dict[str, object]], target: str = "SC-INR", baselines: Sequence[str] = ("LTE", "LIIF")) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        key = (row["seed"], row["dataset"], row["image"], row["scale"], row["scale_num"], row["split"])
        by_key.setdefault(key, {})[str(row["model"])] = row

    out: List[Dict[str, object]] = []
    for key, group in sorted(by_key.items()):
        if target not in group:
            continue
        seed, dataset, image, scale, scale_num, split = key
        target_psnr = finite_float(group[target]["psnr_y"])
        for baseline in baselines:
            if baseline not in group:
                continue
            baseline_psnr = finite_float(group[baseline]["psnr_y"])
            out.append({
                "seed": seed,
                "dataset": dataset,
                "image": image,
                "scale": scale,
                "scale_num": scale_num,
                "split": split,
                "target": target,
                "baseline": baseline,
                "target_psnr_y": f"{target_psnr:.8f}",
                "baseline_psnr_y": f"{baseline_psnr:.8f}",
                "delta_psnr_y": f"{target_psnr - baseline_psnr:.8f}",
                "win": int(target_psnr > baseline_psnr),
            })
    return out


def summarize_delta_rows(rows: List[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    out: List[Dict[str, object]] = []
    for key, items in sorted(groups.items()):
        vals = [finite_float(r["delta_psnr_y"]) for r in items]
        wins = [int(r["win"]) for r in items]
        rec = {k: v for k, v in zip(keys, key)}
        rec.update({
            "n": len(vals),
            "mean_delta": f"{mean(vals):.8f}",
            "median_delta": f"{median(vals):.8f}",
            "std_delta": f"{std(vals):.8f}",
            "sem_delta": f"{sem(vals):.8f}",
            "q05_delta": f"{quantile(vals, 0.05):.8f}",
            "q25_delta": f"{quantile(vals, 0.25):.8f}",
            "q75_delta": f"{quantile(vals, 0.75):.8f}",
            "q95_delta": f"{quantile(vals, 0.95):.8f}",
            "win_rate": f"{mean(wins):.8f}",
        })
        out.append(rec)
    return out


def bootstrap_ci(rows: List[Dict[str, object]], n_boot: int, seed: int) -> List[Dict[str, object]]:
    """Cluster bootstrap over seed and image groups for OOD/ALL delta."""

    rng = np.random.default_rng(seed)
    out: List[Dict[str, object]] = []
    for baseline in sorted({r["baseline"] for r in rows}):
        sub = [r for r in rows if r["baseline"] == baseline]
        for split_name, split_filter in [("ALL", lambda r: True), ("OOD", lambda r: r["split"] == "OOD"), ("ID", lambda r: r["split"] == "ID")]:
            items = [r for r in sub if split_filter(r)]
            if not items:
                continue
            clusters: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
            for row in items:
                clusters.setdefault((str(row["seed"]), str(row["dataset"]), str(row["image"])), []).append(row)
            cluster_keys = list(clusters)
            observed_vals = [finite_float(r["delta_psnr_y"]) for r in items]
            boot_means: List[float] = []
            boot_win_rates: List[float] = []
            for _ in range(n_boot):
                sampled = rng.choice(len(cluster_keys), size=len(cluster_keys), replace=True)
                sample_rows: List[Dict[str, object]] = []
                for idx in sampled:
                    sample_rows.extend(clusters[cluster_keys[int(idx)]])
                vals = [finite_float(r["delta_psnr_y"]) for r in sample_rows]
                wins = [1.0 if v > 0 else 0.0 for v in vals if math.isfinite(v)]
                boot_means.append(mean(vals))
                boot_win_rates.append(mean(wins))
            out.append({
                "baseline": baseline,
                "split": split_name,
                "n_rows": len(items),
                "n_clusters": len(cluster_keys),
                "observed_mean_delta": f"{mean(observed_vals):.8f}",
                "observed_median_delta": f"{median(observed_vals):.8f}",
                "observed_win_rate": f"{mean([1.0 if v > 0 else 0.0 for v in observed_vals]):.8f}",
                "bootstrap_mean_ci_low": f"{quantile(boot_means, 0.025):.8f}",
                "bootstrap_mean_ci_high": f"{quantile(boot_means, 0.975):.8f}",
                "bootstrap_win_rate_ci_low": f"{quantile(boot_win_rates, 0.025):.8f}",
                "bootstrap_win_rate_ci_high": f"{quantile(boot_win_rates, 0.975):.8f}",
            })
    return out


def summarize_per_image(rows: List[Dict[str, object]], out_dir: Path, n_boot: int, seed: int,
                        make_internal_plots: bool = False) -> None:
    deltas = paired_delta_rows(rows)
    write_csv(out_dir / "paired_delta_per_image_scale.csv", deltas)
    write_csv(out_dir / "paired_delta_summary_by_dataset_scale.csv", summarize_delta_rows(deltas, ["baseline", "dataset", "scale", "split"]))
    write_csv(out_dir / "paired_delta_summary_by_scale.csv", summarize_delta_rows(deltas, ["baseline", "scale", "split"]))
    write_csv(out_dir / "paired_delta_summary_by_dataset.csv", summarize_delta_rows(deltas, ["baseline", "dataset", "split"]))
    write_csv(out_dir / "paired_delta_summary_overall.csv", summarize_delta_rows(deltas, ["baseline", "split"]))
    write_csv(out_dir / "paired_delta_bootstrap_ci.csv", bootstrap_ci(deltas, n_boot, seed))

    image_groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in deltas:
        image_groups.setdefault((row["baseline"], row["seed"], row["dataset"], row["image"], row["split"]), []).append(row)
    image_rows: List[Dict[str, object]] = []
    for key, items in sorted(image_groups.items()):
        baseline, seed_id, dataset, image, split = key
        vals = [finite_float(r["delta_psnr_y"]) for r in items]
        image_rows.append({
            "baseline": baseline,
            "seed": seed_id,
            "dataset": dataset,
            "image": image,
            "split": split,
            "n_scales": len(items),
            "mean_delta": f"{mean(vals):.8f}",
            "median_delta": f"{median(vals):.8f}",
            "win_rate": f"{mean([1.0 if v > 0 else 0.0 for v in vals]):.8f}",
        })
    write_csv(out_dir / "image_win_loss_summary.csv", image_rows)
    if make_internal_plots:
        plot_per_image_summaries(out_dir)


def crop_descriptor(gt: torch.Tensor, y0: int, x0: int, size: int, scale: int) -> Dict[str, float]:
    gt_crop = crop_tensor(gt, y0, x0, size)
    gt_y = shave_valid(rgb_to_y(gt_crop), min(scale, max(0, size // 4)))
    grad = sobel_mag(gt_y)
    var = local_variance(gt_y)
    hp = highpass_rmse(gt_y, torch.zeros_like(gt_y))
    fft_low, fft_mid, fft_high = fft_band_power(gt_y - gt_y.mean())
    return {
        "texture_var": float(var.mean().item()),
        "edge_mag": float(grad.mean().item()),
        "highpass_energy": float(hp),
        "fft_low_power": fft_low,
        "fft_mid_power": fft_mid,
        "fft_high_power": fft_high,
    }


def crop_quality(pred: torch.Tensor, gt: torch.Tensor, y0: int, x0: int, size: int, scale: int) -> Dict[str, float]:
    pred_crop = crop_tensor(pred, y0, x0, size)
    gt_crop = crop_tensor(gt, y0, x0, size)
    pred_y = shave_valid(rgb_to_y(pred_crop), min(scale, max(0, size // 4)))
    gt_y = shave_valid(rgb_to_y(gt_crop), min(scale, max(0, size // 4)))
    diff = pred_y - gt_y
    mse = float(diff.square().mean().item())
    grad = sobel_mag(gt_y)
    var = local_variance(gt_y)
    edge = top_mask(grad, 0.20)
    texture = top_mask(var, 0.20)
    flat = bottom_mask(var, 0.20)
    return {
        # 完全平坦 crop 中 Bicubic 可能与 GT 完全一致。为保证正式 CSV 无 inf，
        # 局部诊断使用足够高的 finite cap；该值只影响极少数完美重建平坦负控。
        "psnr_y": 120.0 if mse <= 1e-12 else psnr_from_mse(mse),
        "rmse_y": math.sqrt(mse),
        "edge_rmse": masked_rmse(diff, edge),
        "texture_rmse": masked_rmse(diff, texture),
        "flat_rmse": masked_rmse(diff, flat),
        "highpass_rmse": highpass_rmse(pred_y, gt_y),
    }


def crop_grid(h: int, w: int, size: int, stride: int, max_crops: int) -> List[Tuple[int, int, int]]:
    size = min(size, h, w)
    stride = max(1, stride)
    ys = list(range(0, max(1, h - size + 1), stride))
    xs = list(range(0, max(1, w - size + 1), stride))
    if ys[-1] != h - size:
        ys.append(h - size)
    if xs[-1] != w - size:
        xs.append(w - size)
    coords = [(y, x, size) for y in ys for x in xs]
    if max_crops > 0 and len(coords) > max_crops:
        # 固定、与模型输出无关的均匀抽样，避免大图支配候选池。
        idxs = np.linspace(0, len(coords) - 1, max_crops).round().astype(int)
        coords = [coords[int(i)] for i in idxs]
    return coords


def bin_by_quantiles(values: Sequence[float], low_q: float = 0.33, high_q: float = 0.67) -> List[str]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    lo = float(np.quantile(arr, low_q))
    hi = float(np.quantile(arr, high_q))
    out = []
    for v in arr:
        if v <= lo:
            out.append("low")
        elif v >= hi:
            out.append("high")
        else:
            out.append("mid")
    return out


def add_bins(rows: List[Dict[str, object]]) -> None:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["image"], row["scale"]), []).append(row)
    for items in groups.values():
        for metric, bin_name in [
            ("texture_var", "texture_bin"),
            ("edge_mag", "edge_bin"),
            ("highpass_energy", "highpass_bin"),
        ]:
            bins = bin_by_quantiles([finite_float(r[metric]) for r in items])
            for row, label in zip(items, bins):
                row[bin_name] = label
        for row in items:
            row["flat_control"] = int(row.get("texture_bin") == "low" and row.get("edge_bin") == "low")


def run_local_crops(args, device: torch.device, out_dir: Path) -> List[Dict[str, object]]:
    descriptor_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    datasets = parse_csv_list(args.local_datasets)
    scales = parse_ints(args.local_scales)
    models = parse_csv_list(args.local_models)

    loaded = {name: load_model_from_seed(name, 1, device) for name in models}
    for dataset in datasets:
        images = image_limit(list_images(dataset, args.local_max_images), args.local_max_images)
        for img_path in tqdm(images, desc=f"local-crops:{dataset}"):
            for scale in scales:
                lr, gt = make_lr_hr(img_path, scale, device)
                h, w = gt.shape[-2:]
                coords = crop_grid(h, w, args.crop_size, args.crop_stride, args.max_crops_per_image_scale)
                preds: Dict[str, torch.Tensor] = {"Bicubic": bicubic_baseline(lr, h, w)}
                for model_name, model in loaded.items():
                    preds[model_name] = predict_image(model, lr, h, w, device, args.eval_bsize)

                for crop_id, (y0, x0, size) in enumerate(coords):
                    desc = crop_descriptor(gt, y0, x0, size, scale)
                    descriptor_rows.append({
                        "dataset": DATASET_DISPLAY.get(dataset, dataset),
                        "image": img_path.name,
                        "scale": f"x{scale}",
                        "scale_num": scale,
                        "split": split_for_scale(scale),
                        "crop_id": crop_id,
                        "crop_y": y0,
                        "crop_x": x0,
                        "crop_size": size,
                        **{k: f"{v:.8f}" for k, v in desc.items()},
                    })
                    for model_name in ["Bicubic"] + models:
                        metrics = crop_quality(preds[model_name], gt, y0, x0, size, scale)
                        metric_rows.append({
                            "dataset": DATASET_DISPLAY.get(dataset, dataset),
                            "image": img_path.name,
                            "scale": f"x{scale}",
                            "scale_num": scale,
                            "split": split_for_scale(scale),
                            "crop_id": crop_id,
                            "crop_y": y0,
                            "crop_x": x0,
                            "crop_size": size,
                            "model": model_name,
                            **{k: f"{v:.8f}" for k, v in metrics.items()},
                        })
                del lr, gt, preds
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    add_bins(descriptor_rows)
    write_csv(out_dir / "local_crop_descriptors.csv", descriptor_rows)

    descriptor_by_key = {
        (r["dataset"], r["image"], r["scale"], str(r["crop_id"])): r
        for r in descriptor_rows
    }
    for row in metric_rows:
        desc = descriptor_by_key[(row["dataset"], row["image"], row["scale"], str(row["crop_id"]))]
        for field in ["texture_var", "edge_mag", "highpass_energy", "texture_bin", "edge_bin", "highpass_bin", "flat_control"]:
            row[field] = desc[field]
    write_csv(out_dir / "local_crop_metrics.csv", metric_rows)
    summarize_local_crops(metric_rows, out_dir, make_internal_plots=args.make_internal_plots)
    if args.render_internal_examples:
        render_representative_examples(args, device, out_dir)
    render_external_advantage_examples(args, device, out_dir)
    return metric_rows


def local_delta_rows(metric_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for row in metric_rows:
        key = (row["dataset"], row["image"], row["scale"], row["scale_num"], row["split"], row["crop_id"], row["crop_y"], row["crop_x"], row["crop_size"])
        by_key.setdefault(key, {})[str(row["model"])] = row
    baselines = ["LIIF", "LTE", "SC-INR-NoPhi", "SC-INR-NoSinc", "Bicubic"]
    out: List[Dict[str, object]] = []
    for key, group in sorted(by_key.items()):
        if "SC-INR" not in group:
            continue
        dataset, image, scale, scale_num, split, crop_id, crop_y, crop_x, crop_size = key
        target = group["SC-INR"]
        for baseline in baselines:
            if baseline not in group:
                continue
            base = group[baseline]
            rec = {
                "dataset": dataset,
                "image": image,
                "scale": scale,
                "scale_num": scale_num,
                "split": split,
                "crop_id": crop_id,
                "crop_y": crop_y,
                "crop_x": crop_x,
                "crop_size": crop_size,
                "baseline": baseline,
                "texture_bin": target["texture_bin"],
                "edge_bin": target["edge_bin"],
                "highpass_bin": target["highpass_bin"],
                "flat_control": target["flat_control"],
                "texture_var": target["texture_var"],
                "edge_mag": target["edge_mag"],
                "highpass_energy": target["highpass_energy"],
            }
            for metric in ["psnr_y", "rmse_y", "edge_rmse", "texture_rmse", "flat_rmse", "highpass_rmse"]:
                t = finite_float(target[metric])
                b = finite_float(base[metric])
                delta = t - b
                if metric.endswith("rmse"):
                    win = int(t < b)
                else:
                    win = int(t > b)
                rec[f"sc_inr_{metric}"] = f"{t:.8f}"
                rec[f"baseline_{metric}"] = f"{b:.8f}"
                rec[f"delta_{metric}"] = f"{delta:.8f}"
                rec[f"win_{metric}"] = win
            out.append(rec)
    return out


def summarize_local_crops(metric_rows: List[Dict[str, object]], out_dir: Path, make_internal_plots: bool = False) -> None:
    deltas = local_delta_rows(metric_rows)
    external_rows = external_advantage_rows(deltas)
    write_csv(out_dir / "external_advantage_per_crop.csv", external_rows)
    for keys, filename in [
        (["scale", "split"], "external_advantage_summary_by_scale.csv"),
        (["scale", "texture_bin", "split"], "external_advantage_summary_by_scale_texture.csv"),
        (["scale", "edge_bin", "split"], "external_advantage_summary_by_scale_edge.csv"),
        (["scale", "highpass_bin", "split"], "external_advantage_summary_by_scale_highpass.csv"),
    ]:
        write_csv(out_dir / filename, summarize_external_advantage(external_rows, keys))
    write_csv(out_dir / "external_advantage_region_pool.csv", external_advantage_region_pool(external_rows))
    if make_internal_plots:
        write_csv(out_dir / "local_crop_paired_deltas.csv", deltas)
        for keys, name in [
            (["baseline", "split"], "local_delta_summary_overall.csv"),
            (["baseline", "scale", "split"], "local_delta_summary_by_scale.csv"),
            (["baseline", "dataset", "split"], "local_delta_summary_by_dataset.csv"),
            (["baseline", "texture_bin", "split"], "local_delta_summary_by_texture_bin.csv"),
            (["baseline", "edge_bin", "split"], "local_delta_summary_by_edge_bin.csv"),
            (["baseline", "highpass_bin", "split"], "local_delta_summary_by_highpass_bin.csv"),
            (["baseline", "flat_control", "split"], "local_delta_summary_by_flat_control.csv"),
            (["baseline", "dataset", "image", "split"], "local_delta_summary_by_image.csv"),
        ]:
            write_csv(out_dir / name, summarize_local_delta(deltas, keys))
        write_csv(out_dir / "representative_crop_pool.csv", representative_pool(deltas))
        summarize_advantage_zones(deltas, out_dir, make_internal_plots=True)
        plot_local_summaries(out_dir)


def summarize_local_delta(rows: List[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, object]] = []
    for key, items in sorted(groups.items()):
        psnr = [finite_float(r["delta_psnr_y"]) for r in items]
        tex = [finite_float(r["delta_texture_rmse"]) for r in items]
        hp = [finite_float(r["delta_highpass_rmse"]) for r in items]
        rec = {k: v for k, v in zip(keys, key)}
        rec.update({
            "n": len(items),
            "mean_delta_psnr_y": f"{mean(psnr):.8f}",
            "median_delta_psnr_y": f"{median(psnr):.8f}",
            "q25_delta_psnr_y": f"{quantile(psnr, 0.25):.8f}",
            "q75_delta_psnr_y": f"{quantile(psnr, 0.75):.8f}",
            "win_rate_psnr_y": f"{mean([1.0 if finite_float(r['delta_psnr_y']) > 0 else 0.0 for r in items]):.8f}",
            "mean_delta_texture_rmse": f"{mean(tex):.8f}",
            "win_rate_texture_rmse": f"{mean([1.0 if finite_float(r['delta_texture_rmse']) < 0 else 0.0 for r in items]):.8f}",
            "mean_delta_highpass_rmse": f"{mean(hp):.8f}",
            "win_rate_highpass_rmse": f"{mean([1.0 if finite_float(r['delta_highpass_rmse']) < 0 else 0.0 for r in items]):.8f}",
        })
        out.append(rec)
    return out


def representative_pool(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    focus = [
        r for r in rows
        if r["baseline"] == "LTE"
        and r["split"] == "OOD"
        and (r["texture_bin"] == "high" or r["highpass_bin"] == "high")
    ]
    if not focus:
        focus = [r for r in rows if r["baseline"] == "LTE" and r["split"] == "OOD"]
    focus = sorted(focus, key=lambda r: finite_float(r["delta_psnr_y"]))
    selected: List[Tuple[str, Dict[str, object]]] = []
    for label, idxs in [
        ("failure", range(0, min(6, len(focus)))),
        ("neutral", [max(0, len(focus) // 2 - 2), max(0, len(focus) // 2 - 1), min(len(focus) - 1, len(focus) // 2), min(len(focus) - 1, len(focus) // 2 + 1)] if focus else []),
        ("top_positive", range(max(0, len(focus) - 8), len(focus))),
    ]:
        for idx in idxs:
            if 0 <= idx < len(focus):
                selected.append((label, focus[idx]))
    out = []
    seen = set()
    for label, row in selected:
        key = (row["dataset"], row["image"], row["scale"], row["crop_id"])
        if key in seen:
            continue
        seen.add(key)
        rec = dict(row)
        rec["representative_type"] = label
        out.append(rec)
    return out


def is_structured_crop(row: Dict[str, object]) -> bool:
    """论文定性候选只看有纹理、边缘或高频结构的区域，避开平坦数值幻觉。"""

    bins = [row["texture_bin"], row["edge_bin"], row["highpass_bin"]]
    if str(row.get("flat_control", "0")) == "1":
        return False
    return ("high" in bins) or (sum(1 for b in bins if b in {"mid", "high"}) >= 2)


def external_advantage_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Build per-crop SC-INR vs LIIF/LTE rows for paper-facing advantage regions."""

    by_crop: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        if row["baseline"] not in {"LIIF", "LTE"}:
            continue
        key = (
            row["dataset"],
            row["image"],
            row["scale"],
            row["scale_num"],
            row["split"],
            row["crop_id"],
            row["crop_y"],
            row["crop_x"],
            row["crop_size"],
            row["texture_bin"],
            row["edge_bin"],
            row["highpass_bin"],
            row["flat_control"],
        )
        by_crop.setdefault(key, {})[str(row["baseline"])] = row

    out: List[Dict[str, object]] = []
    for key, group in sorted(by_crop.items()):
        if not {"LIIF", "LTE"}.issubset(group):
            continue
        (
            dataset,
            image,
            scale,
            scale_num,
            split,
            crop_id,
            crop_y,
            crop_x,
            crop_size,
            texture_bin,
            edge_bin,
            highpass_bin,
            flat_control,
        ) = key
        liif = group["LIIF"]
        lte = group["LTE"]
        delta_liif = finite_float(liif["delta_psnr_y"])
        delta_lte = finite_float(lte["delta_psnr_y"])
        rec = {
            "dataset": dataset,
            "image": image,
            "scale": scale,
            "scale_num": scale_num,
            "split": split,
            "crop_id": crop_id,
            "crop_y": crop_y,
            "crop_x": crop_x,
            "crop_size": crop_size,
            "texture_bin": texture_bin,
            "edge_bin": edge_bin,
            "highpass_bin": highpass_bin,
            "flat_control": flat_control,
            "structured_crop": int(is_structured_crop({
                "texture_bin": texture_bin,
                "edge_bin": edge_bin,
                "highpass_bin": highpass_bin,
                "flat_control": flat_control,
            })),
            "sc_inr_psnr_y": liif["sc_inr_psnr_y"],
            "liif_psnr_y": liif["baseline_psnr_y"],
            "lte_psnr_y": lte["baseline_psnr_y"],
            "delta_vs_liif": f"{delta_liif:.8f}",
            "delta_vs_lte": f"{delta_lte:.8f}",
            "min_external_delta": f"{min(delta_liif, delta_lte):.8f}",
            "mean_external_delta": f"{((delta_liif + delta_lte) / 2.0):.8f}",
            "wins_external": int(delta_liif > 0 and delta_lte > 0),
            "texture_var": liif["texture_var"],
            "edge_mag": liif["edge_mag"],
            "highpass_energy": liif["highpass_energy"],
        }
        out.append(rec)
    return out


def summarize_external_advantage(rows: List[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, object]] = []
    for key, items in sorted(groups.items()):
        liif = [finite_float(r["delta_vs_liif"]) for r in items]
        lte = [finite_float(r["delta_vs_lte"]) for r in items]
        min_ext = [finite_float(r["min_external_delta"]) for r in items]
        rec = {k: v for k, v in zip(keys, key)}
        rec.update({
            "n": len(items),
            "median_liif_delta": f"{median(liif):.8f}",
            "median_lte_delta": f"{median(lte):.8f}",
            "median_min_external_delta": f"{median(min_ext):.8f}",
            "q25_min_external_delta": f"{quantile(min_ext, 0.25):.8f}",
            "q75_min_external_delta": f"{quantile(min_ext, 0.75):.8f}",
            "win_rate_liif": f"{mean([1.0 if v > 0 else 0.0 for v in liif]):.8f}",
            "win_rate_lte": f"{mean([1.0 if v > 0 else 0.0 for v in lte]):.8f}",
            "win_rate_external": f"{mean([finite_float(r['wins_external']) for r in items]):.8f}",
        })
        out.append(rec)
    return out


def external_advantage_region_pool(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Select paper-facing top/neutral/failure candidates against LIIF/LTE only."""

    focus = [
        r for r in rows
        if r["split"] == "OOD"
        and int(r["structured_crop"]) == 1
    ]
    if not focus:
        focus = [r for r in rows if r["split"] == "OOD"]
    sorted_rows = sorted(focus, key=lambda r: finite_float(r["min_external_delta"]))

    def add_with_caps(target: List[Tuple[str, Dict[str, object]]], label: str, candidates: List[Dict[str, object]],
                      limit: int, per_image_cap: int = 2) -> None:
        counts: Dict[Tuple[object, ...], int] = {}
        for row in candidates:
            image_key = (row["dataset"], row["image"])
            if counts.get(image_key, 0) >= per_image_cap:
                continue
            target.append((label, row))
            counts[image_key] = counts.get(image_key, 0) + 1
            if len([x for x in target if x[0] == label]) >= limit:
                break

    selected: List[Tuple[str, Dict[str, object]]] = []
    positives = [r for r in reversed(sorted_rows) if finite_float(r["min_external_delta"]) > 0]
    failures = [r for r in sorted_rows if finite_float(r["min_external_delta"]) < 0]
    neutral = sorted(sorted_rows, key=lambda r: abs(finite_float(r["min_external_delta"])))
    add_with_caps(selected, "external_top_positive", positives, 12)
    add_with_caps(selected, "external_neutral", neutral, 6)
    add_with_caps(selected, "external_failure", failures, 6)

    out: List[Dict[str, object]] = []
    seen = set()
    for label, row in selected:
        key = (row["dataset"], row["image"], row["scale"], row["crop_id"], label)
        if key in seen:
            continue
        seen.add(key)
        rec = dict(row)
        rec["representative_type"] = label
        out.append(rec)
    return out


def summarize_advantage_zones(rows: List[Dict[str, object]], out_dir: Path, make_internal_plots: bool = False) -> None:
    """Summarize zones where full SC-INR beats external baselines and ablations."""

    by_crop: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["image"],
            row["scale"],
            row["scale_num"],
            row["split"],
            row["crop_id"],
            row["crop_y"],
            row["crop_x"],
            row["crop_size"],
            row["texture_bin"],
            row["edge_bin"],
            row["highpass_bin"],
            row["flat_control"],
        )
        by_crop.setdefault(key, {})[str(row["baseline"])] = row

    crop_rows: List[Dict[str, object]] = []
    for key, group in sorted(by_crop.items()):
        if not {"LIIF", "LTE", "SC-INR-NoPhi", "SC-INR-NoSinc"}.issubset(group):
            continue
        (
            dataset,
            image,
            scale,
            scale_num,
            split,
            crop_id,
            crop_y,
            crop_x,
            crop_size,
            texture_bin,
            edge_bin,
            highpass_bin,
            flat_control,
        ) = key
        deltas = {b: finite_float(group[b]["delta_psnr_y"]) for b in ["LIIF", "LTE", "SC-INR-NoPhi", "SC-INR-NoSinc"]}
        rec = {
            "dataset": dataset,
            "image": image,
            "scale": scale,
            "scale_num": scale_num,
            "split": split,
            "crop_id": crop_id,
            "crop_y": crop_y,
            "crop_x": crop_x,
            "crop_size": crop_size,
            "texture_bin": texture_bin,
            "edge_bin": edge_bin,
            "highpass_bin": highpass_bin,
            "flat_control": flat_control,
            "delta_vs_liif": f"{deltas['LIIF']:.8f}",
            "delta_vs_lte": f"{deltas['LTE']:.8f}",
            "delta_vs_nophi": f"{deltas['SC-INR-NoPhi']:.8f}",
            "delta_vs_nosinc": f"{deltas['SC-INR-NoSinc']:.8f}",
            "wins_external": int(deltas["LIIF"] > 0 and deltas["LTE"] > 0),
            "wins_ablation": int(deltas["SC-INR-NoPhi"] > 0 and deltas["SC-INR-NoSinc"] > 0),
            "wins_all": int(all(v > 0 for v in deltas.values())),
            "min_external_delta": f"{min(deltas['LIIF'], deltas['LTE']):.8f}",
            "min_ablation_delta": f"{min(deltas['SC-INR-NoPhi'], deltas['SC-INR-NoSinc']):.8f}",
            "min_all_delta": f"{min(deltas.values()):.8f}",
        }
        crop_rows.append(rec)
    write_csv(out_dir / "advantage_zone_per_crop.csv", crop_rows)
    write_csv(out_dir / "advantage_zone_representative_pool.csv", advantage_zone_representative_pool(crop_rows))

    for keys, filename in [
        (["split"], "advantage_zone_summary_overall.csv"),
        (["scale", "split"], "advantage_zone_summary_by_scale.csv"),
        (["texture_bin", "split"], "advantage_zone_summary_by_texture_bin.csv"),
        (["edge_bin", "split"], "advantage_zone_summary_by_edge_bin.csv"),
        (["highpass_bin", "split"], "advantage_zone_summary_by_highpass_bin.csv"),
        (["scale", "texture_bin", "split"], "advantage_zone_summary_by_scale_texture.csv"),
        (["scale", "highpass_bin", "split"], "advantage_zone_summary_by_scale_highpass.csv"),
        (["flat_control", "split"], "advantage_zone_summary_by_flat_control.csv"),
    ]:
        write_csv(out_dir / filename, summarize_advantage_zone(crop_rows, keys))
    for keys, filename in [
        (["split"], "advantage_margin_summary_overall.csv"),
        (["scale", "split"], "advantage_margin_summary_by_scale.csv"),
        (["texture_bin", "split"], "advantage_margin_summary_by_texture_bin.csv"),
        (["edge_bin", "split"], "advantage_margin_summary_by_edge_bin.csv"),
        (["highpass_bin", "split"], "advantage_margin_summary_by_highpass_bin.csv"),
        (["scale", "texture_bin", "split"], "advantage_margin_summary_by_scale_texture.csv"),
        (["scale", "edge_bin", "split"], "advantage_margin_summary_by_scale_edge.csv"),
        (["scale", "highpass_bin", "split"], "advantage_margin_summary_by_scale_highpass.csv"),
        (["flat_control", "split"], "advantage_margin_summary_by_flat_control.csv"),
    ]:
        write_csv(out_dir / filename, summarize_advantage_margins(crop_rows, keys))
    write_advantage_margin_rankings(crop_rows, out_dir)
    write_advantage_zone_rankings(crop_rows, out_dir)
    if make_internal_plots:
        plot_advantage_zones(out_dir)


def summarize_advantage_zone(rows: List[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, object]] = []
    for key, items in sorted(groups.items()):
        ext = [int(r["wins_external"]) for r in items]
        abl = [int(r["wins_ablation"]) for r in items]
        all_w = [int(r["wins_all"]) for r in items]
        rec = {k: v for k, v in zip(keys, key)}
        rec.update({
            "n": len(items),
            "external_dominance_rate": f"{mean(ext):.8f}",
            "ablation_dominance_rate": f"{mean(abl):.8f}",
            "all_dominance_rate": f"{mean(all_w):.8f}",
            "median_min_external_delta": f"{median([finite_float(r['min_external_delta']) for r in items]):.8f}",
            "median_min_ablation_delta": f"{median([finite_float(r['min_ablation_delta']) for r in items]):.8f}",
            "median_min_all_delta": f"{median([finite_float(r['min_all_delta']) for r in items]):.8f}",
            "q25_min_all_delta": f"{quantile([finite_float(r['min_all_delta']) for r in items], 0.25):.8f}",
            "q75_min_all_delta": f"{quantile([finite_float(r['min_all_delta']) for r in items], 0.75):.8f}",
        })
        out.append(rec)
    return out


def summarize_advantage_margins(rows: List[Dict[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, object]] = []
    fields = [
        ("liif", "delta_vs_liif"),
        ("lte", "delta_vs_lte"),
        ("nophi", "delta_vs_nophi"),
        ("nosinc", "delta_vs_nosinc"),
    ]
    for key, items in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec["n"] = len(items)
        for short, field in fields:
            vals = [finite_float(r[field]) for r in items]
            rec[f"median_{short}_delta"] = f"{median(vals):.8f}"
            rec[f"q25_{short}_delta"] = f"{quantile(vals, 0.25):.8f}"
            rec[f"q75_{short}_delta"] = f"{quantile(vals, 0.75):.8f}"
            rec[f"win_rate_{short}"] = f"{mean([1.0 if v > 0 else 0.0 for v in vals]):.8f}"
        ext_margins = [min(finite_float(r["delta_vs_liif"]), finite_float(r["delta_vs_lte"])) for r in items]
        abl_margins = [min(finite_float(r["delta_vs_nophi"]), finite_float(r["delta_vs_nosinc"])) for r in items]
        all_margins = [min(finite_float(r[f]) for f in ["delta_vs_liif", "delta_vs_lte", "delta_vs_nophi", "delta_vs_nosinc"]) for r in items]
        rec["median_min_external_delta"] = f"{median(ext_margins):.8f}"
        rec["median_min_ablation_delta"] = f"{median(abl_margins):.8f}"
        rec["median_min_all_delta"] = f"{median(all_margins):.8f}"
        rec["joint_positive_median"] = int(all(finite_float(rec[f"median_{short}_delta"]) > 0 for short, _ in fields))
        rec["joint_win_rate_above_half"] = int(all(finite_float(rec[f"win_rate_{short}"]) > 0.5 for short, _ in fields))
        out.append(rec)
    return out


def advantage_zone_representative_pool(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Select full-method examples by min delta against all external/internal controls.

    这个候选池与 `representative_crop_pool.csv` 不同：后者主要服务 SC-INR vs LTE
    的视觉展示；这里要求完整 SC-INR 同时面对 LIIF/LTE/NoPhi/NoSinc，避免把
    `phi(z)+sinc` 的完整方案优势偷换成只赢 LTE 的局部案例。
    """

    focus = [
        r for r in rows
        if r["split"] == "OOD"
        and (r["texture_bin"] == "high" or r["edge_bin"] == "high" or r["highpass_bin"] == "high")
    ]
    if not focus:
        focus = [r for r in rows if r["split"] == "OOD"]
    focus = sorted(focus, key=lambda r: finite_float(r["min_all_delta"]))
    selected: List[Tuple[str, Dict[str, object]]] = []
    for label, idxs in [
        ("full_failure", range(0, min(8, len(focus)))),
        ("full_neutral", [max(0, len(focus) // 2 - 2), max(0, len(focus) // 2 - 1), min(len(focus) - 1, len(focus) // 2), min(len(focus) - 1, len(focus) // 2 + 1)] if focus else []),
        ("full_top_positive", range(max(0, len(focus) - 10), len(focus))),
    ]:
        for idx in idxs:
            if 0 <= idx < len(focus):
                selected.append((label, focus[idx]))
    out = []
    seen = set()
    for label, row in selected:
        key = (row["dataset"], row["image"], row["scale"], row["crop_id"])
        if key in seen:
            continue
        seen.add(key)
        rec = dict(row)
        rec["representative_type"] = label
        out.append(rec)
    return out


def write_advantage_zone_rankings(rows: List[Dict[str, object]], out_dir: Path) -> None:
    """Rank pre-registered zones without selecting crops by model performance first."""

    ranked: List[Dict[str, object]] = []
    zone_specs = [
        ("scale", ["scale", "split"]),
        ("texture", ["texture_bin", "split"]),
        ("edge", ["edge_bin", "split"]),
        ("highpass", ["highpass_bin", "split"]),
        ("flat_control", ["flat_control", "split"]),
        ("scale_texture", ["scale", "texture_bin", "split"]),
        ("scale_edge", ["scale", "edge_bin", "split"]),
        ("scale_highpass", ["scale", "highpass_bin", "split"]),
    ]
    for zone_type, keys in zone_specs:
        for rec in summarize_advantage_zone(rows, keys):
            if rec.get("split") != "OOD":
                continue
            label_parts = [f"{k}={rec[k]}" for k in keys if k != "split"]
            all_rate = finite_float(rec["all_dominance_rate"])
            abl_rate = finite_float(rec["ablation_dominance_rate"])
            ext_rate = finite_float(rec["external_dominance_rate"])
            med_all = finite_float(rec["median_min_all_delta"])
            q75_all = finite_float(rec["q75_min_all_delta"])
            if all_rate >= 0.5 and med_all > 0:
                strength = "strong_full_zone"
            elif all_rate >= 0.3 or q75_all > 0:
                strength = "conditional_full_zone"
            elif ext_rate >= 0.6:
                strength = "external_only_zone"
            else:
                strength = "weak_or_no_full_zone"
            ranked.append({
                "zone_type": zone_type,
                "zone": ";".join(label_parts),
                "n": rec["n"],
                "external_dominance_rate": rec["external_dominance_rate"],
                "ablation_dominance_rate": rec["ablation_dominance_rate"],
                "all_dominance_rate": rec["all_dominance_rate"],
                "median_min_external_delta": rec["median_min_external_delta"],
                "median_min_ablation_delta": rec["median_min_ablation_delta"],
                "median_min_all_delta": rec["median_min_all_delta"],
                "q75_min_all_delta": rec["q75_min_all_delta"],
                "strength": strength,
                "ranking_score": f"{(2.0 * all_rate + abl_rate + ext_rate + max(0.0, q75_all)):.8f}",
            })
    ranked = sorted(
        ranked,
        key=lambda r: (
            finite_float(r["ranking_score"]),
            finite_float(r["all_dominance_rate"]),
            finite_float(r["ablation_dominance_rate"]),
        ),
        reverse=True,
    )
    write_csv(out_dir / "advantage_zone_ranked.csv", ranked)


def write_advantage_margin_rankings(rows: List[Dict[str, object]], out_dir: Path) -> None:
    ranked: List[Dict[str, object]] = []
    zone_specs = [
        ("scale", ["scale", "split"]),
        ("texture", ["texture_bin", "split"]),
        ("edge", ["edge_bin", "split"]),
        ("highpass", ["highpass_bin", "split"]),
        ("flat_control", ["flat_control", "split"]),
        ("scale_texture", ["scale", "texture_bin", "split"]),
        ("scale_edge", ["scale", "edge_bin", "split"]),
        ("scale_highpass", ["scale", "highpass_bin", "split"]),
    ]
    for zone_type, keys in zone_specs:
        for rec in summarize_advantage_margins(rows, keys):
            if rec.get("split") != "OOD":
                continue
            label_parts = [f"{k}={rec[k]}" for k in keys if k != "split"]
            medians = [finite_float(rec[f"median_{short}_delta"]) for short in ["liif", "lte", "nophi", "nosinc"]]
            wins = [finite_float(rec[f"win_rate_{short}"]) for short in ["liif", "lte", "nophi", "nosinc"]]
            min_median = min(medians)
            min_win = min(wins)
            if min_median > 0 and min_win > 0.5:
                strength = "pairwise_margin_supported"
            elif finite_float(rec["median_min_external_delta"]) > 0 and finite_float(rec["median_min_ablation_delta"]) > -0.02:
                strength = "external_supported_ablation_borderline"
            else:
                strength = "not_joint_margin_supported"
            ranked.append({
                "zone_type": zone_type,
                "zone": ";".join(label_parts),
                "n": rec["n"],
                "median_liif_delta": rec["median_liif_delta"],
                "median_lte_delta": rec["median_lte_delta"],
                "median_nophi_delta": rec["median_nophi_delta"],
                "median_nosinc_delta": rec["median_nosinc_delta"],
                "win_rate_liif": rec["win_rate_liif"],
                "win_rate_lte": rec["win_rate_lte"],
                "win_rate_nophi": rec["win_rate_nophi"],
                "win_rate_nosinc": rec["win_rate_nosinc"],
                "median_min_external_delta": rec["median_min_external_delta"],
                "median_min_ablation_delta": rec["median_min_ablation_delta"],
                "median_min_all_delta": rec["median_min_all_delta"],
                "joint_positive_median": rec["joint_positive_median"],
                "joint_win_rate_above_half": rec["joint_win_rate_above_half"],
                "strength": strength,
                "ranking_score": f"{(min_median + 0.05 * min_win + finite_float(rec['median_min_external_delta'])):.8f}",
            })
    ranked = sorted(
        ranked,
        key=lambda r: (
            int(r["joint_positive_median"]),
            int(r["joint_win_rate_above_half"]),
            finite_float(r["ranking_score"]),
        ),
        reverse=True,
    )
    write_csv(out_dir / "advantage_margin_ranked.csv", ranked)


def plot_advantage_zones(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    rows = [r for r in read_csv(out_dir / "advantage_zone_summary_by_scale.csv") if r.get("split") == "OOD"]
    if rows:
        rows = sorted(rows, key=lambda r: float(str(r["scale"]).lstrip("x")))
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(8, 4.2))
        width = 0.25
        for offset, field, label, color in [
            (-width, "external_dominance_rate", "win LIIF+LTE", "#187C83"),
            (0.0, "ablation_dominance_rate", "win NoPhi+NoSinc", "#8172B2"),
            (width, "all_dominance_rate", "win all four", "#B22222"),
        ]:
            ax.bar(x + offset, [finite_float(r[field]) for r in rows], width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([r["scale"] for r in rows])
        ax.set_ylim(0, 1)
        ax.set_ylabel("dominance rate")
        ax.set_title("OOD local dominance by scale")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "advantage_zone_dominance_by_scale.png", dpi=200)
        plt.close(fig)

    for source, xfield, filename, title in [
        ("advantage_zone_summary_by_texture_bin.csv", "texture_bin", "advantage_zone_by_texture_bin.png", "OOD dominance by texture bin"),
        ("advantage_zone_summary_by_highpass_bin.csv", "highpass_bin", "advantage_zone_by_highpass_bin.png", "OOD dominance by highpass bin"),
        ("advantage_zone_summary_by_flat_control.csv", "flat_control", "advantage_zone_by_flat_control.png", "OOD dominance by flat-control bin"),
    ]:
        rows = [r for r in read_csv(out_dir / source) if r.get("split") == "OOD"]
        if not rows:
            continue
        labels = sorted({r[xfield] for r in rows})
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        width = 0.25
        for offset, field, label, color in [
            (-width, "external_dominance_rate", "win LIIF+LTE", "#187C83"),
            (0.0, "ablation_dominance_rate", "win NoPhi+NoSinc", "#8172B2"),
            (width, "all_dominance_rate", "win all four", "#B22222"),
        ]:
            vals = []
            for bin_label in labels:
                match = [r for r in rows if r[xfield] == bin_label]
                vals.append(finite_float(match[0][field]) if match else np.nan)
            ax.bar(x + offset, vals, width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("dominance rate")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=200)
        plt.close(fig)
    plot_advantage_zone_heatmaps(out_dir)
    plot_advantage_margin_heatmaps(out_dir)
    plot_min_all_delta_boxplots(out_dir)


def plot_advantage_zone_heatmaps(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    for source, yfield, filename, title in [
        ("advantage_zone_summary_by_scale_texture.csv", "texture_bin", "advantage_zone_heatmap_scale_texture.png", "Scale x texture dominance zones"),
        ("advantage_zone_summary_by_scale_edge.csv", "edge_bin", "advantage_zone_heatmap_scale_edge.png", "Scale x edge dominance zones"),
        ("advantage_zone_summary_by_scale_highpass.csv", "highpass_bin", "advantage_zone_heatmap_scale_highpass.png", "Scale x highpass dominance zones"),
    ]:
        path = out_dir / source
        if not path.exists() and yfield == "edge_bin":
            write_csv(path, summarize_advantage_zone(read_csv(out_dir / "advantage_zone_per_crop.csv"), ["scale", "edge_bin", "split"]))
        rows = [r for r in read_csv(path) if r.get("split") == "OOD"]
        if not rows:
            continue
        scales = sorted({r["scale"] for r in rows}, key=lambda x: float(str(x).lstrip("x")))
        bins = [b for b in ["low", "mid", "high"] if b in {r[yfield] for r in rows}]
        fields = [
            ("external_dominance_rate", "win LIIF+LTE"),
            ("ablation_dominance_rate", "win NoPhi+NoSinc"),
            ("all_dominance_rate", "win all four"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
        for ax, (field, subtitle) in zip(axes, fields):
            mat = np.full((len(bins), len(scales)), np.nan, dtype=np.float64)
            for i, bin_label in enumerate(bins):
                for j, scale in enumerate(scales):
                    match = [r for r in rows if r["scale"] == scale and r[yfield] == bin_label]
                    if match:
                        mat[i, j] = finite_float(match[0][field])
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
            ax.set_title(subtitle, fontsize=9)
            ax.set_xticks(np.arange(len(scales)))
            ax.set_xticklabels(scales)
            ax.set_yticks(np.arange(len(bins)))
            ax.set_yticklabels(bins)
            for i in range(len(bins)):
                for j in range(len(scales)):
                    if math.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8, color="#111111")
        fig.suptitle(title + " (OOD, full candidate pool)", fontsize=10)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="dominance rate")
        fig.savefig(fig_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_advantage_margin_heatmaps(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    for source, yfield, filename, title in [
        ("advantage_margin_summary_by_scale_texture.csv", "texture_bin", "advantage_margin_heatmap_scale_texture.png", "Scale x texture joint margins"),
        ("advantage_margin_summary_by_scale_edge.csv", "edge_bin", "advantage_margin_heatmap_scale_edge.png", "Scale x edge joint margins"),
        ("advantage_margin_summary_by_scale_highpass.csv", "highpass_bin", "advantage_margin_heatmap_scale_highpass.png", "Scale x highpass joint margins"),
    ]:
        rows = [r for r in read_csv(out_dir / source) if r.get("split") == "OOD"]
        if not rows:
            continue
        scales = sorted({r["scale"] for r in rows}, key=lambda x: float(str(x).lstrip("x")))
        bins = [b for b in ["low", "mid", "high"] if b in {r[yfield] for r in rows}]
        fields = [
            ("median_liif_delta", "vs LIIF"),
            ("median_lte_delta", "vs LTE"),
            ("median_nophi_delta", "vs NoPhi"),
            ("median_nosinc_delta", "vs NoSinc"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=True)
        im = None
        for ax, (field, subtitle) in zip(axes, fields):
            mat = np.full((len(bins), len(scales)), np.nan, dtype=np.float64)
            for i, bin_label in enumerate(bins):
                for j, scale in enumerate(scales):
                    match = [r for r in rows if r["scale"] == scale and r[yfield] == bin_label]
                    if match:
                        mat[i, j] = finite_float(match[0][field])
            im = ax.imshow(mat, vmin=-0.08, vmax=0.08, cmap="coolwarm", aspect="auto")
            ax.set_title(subtitle, fontsize=9)
            ax.set_xticks(np.arange(len(scales)))
            ax.set_xticklabels(scales)
            ax.set_yticks(np.arange(len(bins)))
            ax.set_yticklabels(bins)
            for i in range(len(bins)):
                for j in range(len(scales)):
                    if math.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=7, color="#111111")
        fig.suptitle(title + " (median ΔPSNR-Y, OOD)", fontsize=10)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="median ΔPSNR-Y")
        fig.savefig(fig_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_min_all_delta_boxplots(out_dir: Path) -> None:
    rows = [r for r in read_csv(out_dir / "advantage_zone_per_crop.csv") if r.get("split") == "OOD"]
    if not rows:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    scales = sorted({r["scale"] for r in rows}, key=lambda x: float(str(x).lstrip("x")))
    data = [[finite_float(r["min_all_delta"]) for r in rows if r["scale"] == scale] for scale in scales]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.axhline(0, color="#444444", linewidth=1)
    ax.boxplot(data, tick_labels=scales, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#D8E7F3", "edgecolor": "#333333"},
               medianprops={"color": "#B22222", "linewidth": 1.5})
    ax.set_ylabel("min ΔPSNR-Y vs LIIF/LTE/NoPhi/NoSinc")
    ax.set_title("Full-method margin distribution by OOD scale")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "advantage_zone_min_all_delta_box_by_scale.png", dpi=220)
    plt.close(fig)


def render_representative_examples(args, device: torch.device, out_dir: Path) -> None:
    pool_path = out_dir / "representative_crop_pool.csv"
    rows = read_csv(pool_path)
    if not rows:
        return
    by_type: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_type.setdefault(row["representative_type"], []).append(row)
    selected = []
    for label, limit in [("top_positive", args.render_top_k), ("neutral", args.render_neutral_k), ("failure", args.render_failure_k)]:
        selected.extend(by_type.get(label, [])[-limit:] if label == "top_positive" else by_type.get(label, [])[:limit])
    if not selected:
        return
    fig_dir = out_dir / "figures" / "representative_crops"
    ensure_dir(fig_dir)
    loaded = {name: load_model_from_seed(name, 1, device) for name in LOCAL_MODELS}
    for idx, row in enumerate(selected):
        dataset_key = {v: k for k, v in DATASET_DISPLAY.items()}.get(row["dataset"], row["dataset"].lower())
        img_path = DATASETS[dataset_key] / row["image"]
        scale = int(float(row["scale_num"]))
        y0 = int(float(row["crop_y"]))
        x0 = int(float(row["crop_x"]))
        size = int(float(row["crop_size"]))
        lr, gt = make_lr_hr(img_path, scale, device)
        h, w = gt.shape[-2:]
        preds: Dict[str, torch.Tensor] = {"GT": gt, "Bicubic": bicubic_baseline(lr, h, w)}
        for model_name, model in loaded.items():
            preds[model_name] = predict_image(model, lr, h, w, device, args.eval_bsize)
        methods = ["GT", "Bicubic", "LIIF", "LTE", "SC-INR-NoPhi", "SC-INR-NoSinc", "SC-INR"]
        draw_crop_panel(
            preds,
            gt,
            methods,
            y0,
            x0,
            size,
            scale,
            fig_dir / f"{idx:02d}_{row['representative_type']}_{row['dataset']}_{Path(row['image']).stem}_{row['scale']}_crop{row['crop_id']}.png",
            f"{row['representative_type']} {row['dataset']}/{row['image']} {row['scale']} crop=({y0},{x0},{size}) ΔPSNR SC-LTE={finite_float(row['delta_psnr_y']):.3f}",
        )
        del lr, gt, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()


def render_external_advantage_examples(args, device: torch.device, out_dir: Path) -> None:
    pool_path = out_dir / "external_advantage_region_pool.csv"
    rows = read_csv(pool_path)
    if not rows:
        return
    by_type: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_type.setdefault(row["representative_type"], []).append(row)
    selected: List[Dict[str, str]] = []
    for label, limit in [
        ("external_top_positive", args.render_top_k),
        ("external_neutral", args.render_neutral_k),
        ("external_failure", args.render_failure_k),
    ]:
        selected.extend(by_type.get(label, [])[:limit])
    if not selected:
        return
    fig_dir = out_dir / "figures" / "external_advantage_crops"
    ensure_dir(fig_dir)
    loaded = {name: load_model_from_seed(name, 1, device) for name in PAPER_EXTERNAL_MODELS}
    for idx, row in enumerate(selected):
        dataset_key = {v: k for k, v in DATASET_DISPLAY.items()}.get(row["dataset"], row["dataset"].lower())
        img_path = DATASETS[dataset_key] / row["image"]
        scale = int(float(row["scale_num"]))
        y0 = int(float(row["crop_y"]))
        x0 = int(float(row["crop_x"]))
        size = int(float(row["crop_size"]))
        lr, gt = make_lr_hr(img_path, scale, device)
        h, w = gt.shape[-2:]
        preds: Dict[str, torch.Tensor] = {"GT": gt, "Bicubic": bicubic_baseline(lr, h, w)}
        for model_name, model in loaded.items():
            preds[model_name] = predict_image(model, lr, h, w, device, args.eval_bsize)
        methods = ["GT", "Bicubic", "LIIF", "LTE", "SC-INR"]
        title = (
            f"{row['representative_type']} {row['dataset']}/{row['image']} {row['scale']} "
            f"crop=({y0},{x0},{size}) minΔ={finite_float(row['min_external_delta']):.3f}dB"
        )
        draw_clean_crop_panel(
            preds,
            gt,
            methods,
            y0,
            x0,
            size,
            scale,
            fig_dir / f"{idx:02d}_{row['representative_type']}_{row['dataset']}_{Path(row['image']).stem}_{row['scale']}_crop{row['crop_id']}_clean.png",
            title,
        )
        draw_crop_panel(
            preds,
            gt,
            methods,
            y0,
            x0,
            size,
            scale,
            fig_dir / f"{idx:02d}_{row['representative_type']}_{row['dataset']}_{Path(row['image']).stem}_{row['scale']}_crop{row['crop_id']}.png",
            title,
        )
        del lr, gt, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()


def accepted_external_rows(out_dir: Path) -> List[Dict[str, str]]:
    """从用户保留下来的 external crop 文件反查候选池记录。

    这样不会重新自由搜索 top crop；context+zoom 图只围绕已经筛选过的案例生成。
    """

    pool = read_csv(out_dir / "external_advantage_region_pool.csv")
    lookup = {
        (row["dataset"], Path(row["image"]).stem, row["scale"], str(row["crop_id"])): row
        for row in pool
    }
    fig_dir = out_dir / "figures" / "external_advantage_crops"
    pattern = re.compile(r"^\d+_external_(?:top_positive|neutral|failure)_(?P<dataset>[^_]+)_(?P<stem>.+)_(?P<scale>x\d+)_crop(?P<crop>\d+)(?:_clean)?\.png$")
    selected: List[Dict[str, str]] = []
    seen = set()
    for path in sorted(fig_dir.glob("*.png")):
        match = pattern.match(path.name)
        if not match:
            continue
        key = (match.group("dataset"), match.group("stem"), match.group("scale"), match.group("crop"))
        if key in seen or key not in lookup:
            continue
        seen.add(key)
        selected.append(lookup[key])
    return selected


def render_context_zoom_examples(args, device: torch.device, out_dir: Path) -> None:
    rows = accepted_external_rows(out_dir)
    if not rows:
        return
    fig_dir = out_dir / "figures" / "external_advantage_context_zoom"
    ensure_dir(fig_dir)
    loaded = {name: load_model_from_seed(name, 1, device) for name in PAPER_EXTERNAL_MODELS}
    for idx, row in enumerate(rows):
        dataset_key = {v: k for k, v in DATASET_DISPLAY.items()}.get(row["dataset"], row["dataset"].lower())
        img_path = DATASETS[dataset_key] / row["image"]
        scale = int(float(row["scale_num"]))
        y0 = int(float(row["crop_y"]))
        x0 = int(float(row["crop_x"]))
        size = int(float(row["crop_size"]))
        lr, gt = make_lr_hr(img_path, scale, device)
        h, w = gt.shape[-2:]
        preds: Dict[str, torch.Tensor] = {"GT": gt, "Bicubic": bicubic_baseline(lr, h, w)}
        for model_name, model in loaded.items():
            preds[model_name] = predict_image(model, lr, h, w, device, args.eval_bsize)
        methods = ["GT", "Bicubic", "LIIF", "LTE", "SC-INR"]
        title = (
            f"{row['dataset']}/{row['image']} {row['scale']} "
            f"crop=({y0},{x0},{size}) minΔ={finite_float(row['min_external_delta']):.3f}dB"
        )
        draw_context_zoom_panel(
            preds,
            gt,
            methods,
            y0,
            x0,
            size,
            scale,
            args.context_factor,
            fig_dir / f"{idx:02d}_{row['representative_type']}_{row['dataset']}_{Path(row['image']).stem}_{row['scale']}_crop{row['crop_id']}_context_zoom.png",
            title,
        )
        del lr, gt, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()


def x4_sanity_rows(out_dir: Path, limit: int) -> List[Dict[str, str]]:
    rows = [
        r for r in read_csv(out_dir / "external_advantage_per_crop.csv")
        if r.get("scale") == "x4" and r.get("structured_crop") == "1"
    ]
    rows = sorted(rows, key=lambda r: abs(finite_float(r["min_external_delta"])))
    selected: List[Dict[str, str]] = []
    seen_images = set()
    for row in rows:
        image_key = (row["dataset"], row["image"])
        if image_key in seen_images:
            continue
        rec = dict(row)
        rec["representative_type"] = "x4_sanity_neutral"
        selected.append(rec)
        seen_images.add(image_key)
        if len(selected) >= limit:
            break
    return selected


def render_x4_sanity_examples(args, device: torch.device, out_dir: Path) -> None:
    rows = x4_sanity_rows(out_dir, args.render_x4_k)
    if not rows:
        return
    write_csv(out_dir / "external_advantage_x4_sanity_pool.csv", rows)
    fig_dir = out_dir / "figures" / "external_advantage_x4_sanity"
    ensure_dir(fig_dir)
    loaded = {name: load_model_from_seed(name, 1, device) for name in PAPER_EXTERNAL_MODELS}
    for idx, row in enumerate(rows):
        dataset_key = {v: k for k, v in DATASET_DISPLAY.items()}.get(row["dataset"], row["dataset"].lower())
        img_path = DATASETS[dataset_key] / row["image"]
        scale = int(float(row["scale_num"]))
        y0 = int(float(row["crop_y"]))
        x0 = int(float(row["crop_x"]))
        size = int(float(row["crop_size"]))
        lr, gt = make_lr_hr(img_path, scale, device)
        h, w = gt.shape[-2:]
        preds: Dict[str, torch.Tensor] = {"GT": gt, "Bicubic": bicubic_baseline(lr, h, w)}
        for model_name, model in loaded.items():
            preds[model_name] = predict_image(model, lr, h, w, device, args.eval_bsize)
        methods = ["GT", "Bicubic", "LIIF", "LTE", "SC-INR"]
        title = (
            f"x4 sanity {row['dataset']}/{row['image']} crop=({y0},{x0},{size}) "
            f"minΔ={finite_float(row['min_external_delta']):.3f}dB"
        )
        draw_clean_crop_panel(
            preds,
            gt,
            methods,
            y0,
            x0,
            size,
            scale,
            fig_dir / f"{idx:02d}_x4_sanity_{row['dataset']}_{Path(row['image']).stem}_crop{row['crop_id']}_clean.png",
            title,
        )
        draw_context_zoom_panel(
            preds,
            gt,
            methods,
            y0,
            x0,
            size,
            scale,
            args.context_factor,
            fig_dir / f"{idx:02d}_x4_sanity_{row['dataset']}_{Path(row['image']).stem}_crop{row['crop_id']}_context_zoom.png",
            title,
        )
        del lr, gt, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()


def context_box(y0: int, x0: int, size: int, h: int, w: int, factor: int) -> Tuple[int, int, int]:
    csize = min(max(size, size * max(1, factor)), h, w)
    cy = y0 + size // 2
    cx = x0 + size // 2
    y = min(max(0, cy - csize // 2), h - csize)
    x = min(max(0, cx - csize // 2), w - csize)
    return int(y), int(x), int(csize)


def draw_context_zoom_panel(preds: Dict[str, torch.Tensor], gt: torch.Tensor, methods: Sequence[str],
                            y0: int, x0: int, size: int, scale: int, context_factor: int,
                            out_path: Path, title: str) -> None:
    """上下文 + zoom 图：上排看更大局部，下排看原始 crop。"""

    ensure_dir(out_path.parent)
    h, w = gt.shape[-2:]
    cy, cx, csize = context_box(y0, x0, size, h, w, context_factor)
    n = len(methods)
    fig = plt.figure(figsize=(2.05 * n + 2.3, 5.1))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1.15] + [1] * n)
    full_ax = fig.add_subplot(gs[:, 0])
    full_ax.imshow(to_numpy_img(gt))
    full_ax.add_patch(Rectangle((cx, cy), csize, csize, fill=False, edgecolor="#1f77b4", linewidth=1.8))
    full_ax.add_patch(Rectangle((x0, y0), size, size, fill=False, edgecolor="#d62728", linewidth=2.0))
    full_ax.axis("off")
    full_ax.set_title("location", fontsize=9)
    for col, name in enumerate(methods):
        pred_context = crop_tensor(preds[name], cy, cx, csize)
        pred_zoom = crop_tensor(preds[name], y0, x0, size)
        ax_ctx = fig.add_subplot(gs[0, col + 1])
        ax_ctx.imshow(to_numpy_img(pred_context))
        ax_ctx.axis("off")
        ax_ctx.set_title(name, fontsize=9)
        ax_zoom = fig.add_subplot(gs[1, col + 1])
        ax_zoom.imshow(to_numpy_img(pred_zoom))
        ax_zoom.axis("off")
        label = "zoom"
        if name != "GT":
            m = crop_quality(preds[name], gt, y0, x0, size, scale)
            label += f"\n{m['psnr_y']:.2f} dB"
        ax_zoom.set_title(label, fontsize=8)
    fig.suptitle(title, fontsize=9, y=0.985)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_clean_crop_panel(preds: Dict[str, torch.Tensor], gt: torch.Tensor, methods: Sequence[str],
                          y0: int, x0: int, size: int, scale: int, out_path: Path, title: str) -> None:
    """论文主图候选：只展示 crop 本身，不用热图/柱状图抢注意力。"""

    ensure_dir(out_path.parent)
    n = len(methods)
    fig = plt.figure(figsize=(2.05 * n + 2.0, 2.95))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1.12] + [1] * n)
    full_ax = fig.add_subplot(gs[0, 0])
    full_ax.imshow(to_numpy_img(gt))
    full_ax.add_patch(Rectangle((x0, y0), size, size, fill=False, edgecolor="#d62728", linewidth=2.0))
    full_ax.axis("off")
    full_ax.set_title("location", fontsize=9)
    for col, name in enumerate(methods):
        pred_crop = crop_tensor(preds[name], y0, x0, size)
        ax = fig.add_subplot(gs[0, col + 1])
        ax.imshow(to_numpy_img(pred_crop))
        ax.axis("off")
        label = name
        if name != "GT":
            m = crop_quality(preds[name], gt, y0, x0, size, scale)
            label += f"\n{m['psnr_y']:.2f} dB"
        ax.set_title(label, fontsize=9)
    fig.suptitle(title, fontsize=9, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_crop_panel(preds: Dict[str, torch.Tensor], gt: torch.Tensor, methods: Sequence[str],
                    y0: int, x0: int, size: int, scale: int, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    n = len(methods)
    fig = plt.figure(figsize=(2.0 * n + 2.5, 4.7))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1.35] + [1] * n)
    full_ax = fig.add_subplot(gs[:, 0])
    full_ax.imshow(to_numpy_img(gt))
    full_ax.add_patch(Rectangle((x0, y0), size, size, fill=False, edgecolor="#d62728", linewidth=2.0))
    full_ax.axis("off")
    full_ax.set_title("GT full", fontsize=8)
    gt_crop = crop_tensor(gt, y0, x0, size)
    for col, name in enumerate(methods):
        pred_crop = crop_tensor(preds[name], y0, x0, size)
        ax = fig.add_subplot(gs[0, col + 1])
        ax.imshow(to_numpy_img(pred_crop))
        ax.axis("off")
        label = name
        if name != "GT":
            m = crop_quality(preds[name], gt, y0, x0, size, scale)
            label += f"\n{m['psnr_y']:.2f}dB"
        ax.set_title(label, fontsize=8)
        err_ax = fig.add_subplot(gs[1, col + 1])
        if name == "GT":
            err = np.zeros((size, size), dtype=np.float32)
        else:
            err = torch.abs(rgb_to_y(pred_crop) - rgb_to_y(gt_crop)).detach().cpu().squeeze().numpy()
        err_ax.imshow(err, cmap="magma", vmin=0, vmax=0.12)
        err_ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_image_summaries(out_dir: Path) -> None:
    rows = read_csv(out_dir / "paired_delta_summary_by_scale.csv")
    if not rows:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    for baseline in sorted({r["baseline"] for r in rows}):
        sub = [r for r in rows if r["baseline"] == baseline]
        scales = sorted(sub, key=lambda r: float(r["scale"].lstrip("x")))
        x = np.arange(len(scales))
        fig, ax1 = plt.subplots(figsize=(9, 4.2))
        ax1.axhline(0, color="#888888", linewidth=1)
        ax1.bar(x - 0.18, [finite_float(r["median_delta"]) for r in scales], 0.36, label="median ΔPSNR", color="#B22222")
        ax1.bar(x + 0.18, [finite_float(r["mean_delta"]) for r in scales], 0.36, label="mean ΔPSNR", color="#DD8452")
        ax1.set_ylabel("SC-INR - baseline PSNR-Y (dB)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([r["scale"] for r in scales])
        ax2 = ax1.twinx()
        ax2.plot(x, [finite_float(r["win_rate"]) for r in scales], color="#187C83", marker="o", label="win-rate")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("win-rate")
        ax1.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"per_scale_delta_vs_{baseline}.png", dpi=200)
        plt.close(fig)


def plot_local_summaries(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    for path, xfield, filename, title in [
        (out_dir / "local_delta_summary_by_texture_bin.csv", "texture_bin", "local_delta_by_texture_bin.png", "Local ΔPSNR by texture bin"),
        (out_dir / "local_delta_summary_by_edge_bin.csv", "edge_bin", "local_delta_by_edge_bin.png", "Local ΔPSNR by edge bin"),
        (out_dir / "local_delta_summary_by_highpass_bin.csv", "highpass_bin", "local_delta_by_highpass_bin.png", "Local ΔPSNR by highpass bin"),
        (out_dir / "local_delta_summary_by_flat_control.csv", "flat_control", "local_delta_by_flat_control.png", "Local ΔPSNR by flat control"),
    ]:
        rows = [r for r in read_csv(path) if r.get("baseline") in {"LTE", "LIIF", "SC-INR-NoPhi", "SC-INR-NoSinc"} and r.get("split") == "OOD"]
        if not rows:
            continue
        xs = sorted({r[xfield] for r in rows})
        baselines = ["LIIF", "LTE", "SC-INR-NoPhi", "SC-INR-NoSinc"]
        x = np.arange(len(xs))
        width = 0.8 / len(baselines)
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.axhline(0, color="#888888", linewidth=1)
        for i, baseline in enumerate(baselines):
            vals = []
            for label in xs:
                match = [r for r in rows if r[xfield] == label and r["baseline"] == baseline]
                vals.append(finite_float(match[0]["median_delta_psnr_y"]) if match else np.nan)
            ax.bar(x + (i - len(baselines) / 2) * width + width / 2, vals, width, label=baseline, color=STYLE.get(baseline, "#888888"))
        ax.set_xticks(x)
        ax.set_xticklabels(xs)
        ax.set_ylabel("median local ΔPSNR-Y (SC-INR - baseline)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=200)
        plt.close(fig)


def regression_against_canonical(out_dir: Path) -> List[Dict[str, object]]:
    per_image = read_csv(out_dir / "per_image_per_scale_metrics.csv")
    canonical = read_csv(ROOT / "artifacts" / "derived" / "benchmarks" / "benchmark_all_models_long.csv")
    if not per_image or not canonical:
        return []
    groups: Dict[Tuple[str, str, str, str], List[float]] = {}
    for row in per_image:
        key = (str(row["seed"]), row["model"], row["dataset"], row["scale"])
        groups.setdefault(key, []).append(finite_float(row["psnr_y"]))
    canon_lookup: Dict[Tuple[str, str, str, str], float] = {}
    for row in canonical:
        if row["canonical_model"] in MAIN_MODELS:
            canon_lookup[(row["seed"], row["canonical_model"], row["dataset"], row["scale"])] = finite_float(row["psnr"])
    out = []
    for key, vals in sorted(groups.items()):
        calc = mean(vals)
        canon = canon_lookup.get(key, float("nan"))
        out.append({
            "seed": key[0],
            "model": key[1],
            "dataset": key[2],
            "scale": key[3],
            "n": len(vals),
            "computed_mean_psnr_y": f"{calc:.8f}",
            "canonical_psnr": f"{canon:.8f}" if math.isfinite(canon) else "nan",
            "abs_diff": f"{abs(calc - canon):.8f}" if math.isfinite(canon) else "nan",
        })
    write_csv(out_dir / "canonical_regression_check.csv", out)
    return out


def write_readme(out_dir: Path, args, per_image_rows: List[Dict[str, object]] | None, local_rows: List[Dict[str, object]] | None) -> None:
    paired = read_csv(out_dir / "paired_delta_bootstrap_ci.csv")
    local_overall = read_csv(out_dir / "local_delta_summary_overall.csv")
    external_scale = read_csv(out_dir / "external_advantage_summary_by_scale.csv")
    external_pool = read_csv(out_dir / "external_advantage_region_pool.csv")
    x4_pool = read_csv(out_dir / "external_advantage_x4_sanity_pool.csv")
    lines = [
        "# SC-INR 证据链补强诊断 2026-05-21",
        "",
        "本目录当前主用途是定位 `SC-INR` 相对外部基线 `LIIF/LTE` 的可视优势区间。",
        "消融相关诊断默认不再生成；如需内部审查，可显式开启 `--make_internal_plots`。",
        "",
        "## 协议",
        "",
        f"- 逐图统计：models `{args.per_image_models}`，seeds `{args.seeds}`，datasets `{args.per_image_datasets}`，scales `{args.per_image_scales}`。",
        f"- 局部候选池：models `Bicubic,{args.local_models}`，datasets `{args.local_datasets}`，scales `{args.local_scales}`。",
        f"- crop 规则：固定 `{args.crop_size}x{args.crop_size}`，stride `{args.crop_stride}`，每图每尺度最多 `{args.max_crops_per_image_scale}` 个 crop。",
        "- crop 分层只依赖 GT：texture local variance、Sobel edge、highpass energy；不按模型胜负筛选。",
        "- 论文候选图只从 OOD 且非平坦结构 crop 中选择，排序指标为 `min(SC-INR-LIIF, SC-INR-LTE)`。",
        "- 主图只展示 `GT/Bicubic/LIIF/LTE/SC-INR`；NoPhi/NoSinc 留作后续单独消融讨论。",
        "- bootstrap CI 按 seed+dataset+image 聚类，避免把同一图多个 scale 当完全独立样本。",
        "",
        "## 关键结果摘要",
        "",
    ]
    if paired:
        lines.append("### 3-seed paired statistics")
        lines.append("")
        lines.append("| Baseline | Split | Mean Δ | Median Δ | Win-rate | 95% CI mean |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in paired:
            if row["split"] in {"OOD", "ALL"}:
                lines.append(
                    f"| {row['baseline']} | {row['split']} | {float(row['observed_mean_delta']):.4f} | "
                    f"{float(row['observed_median_delta']):.4f} | {float(row['observed_win_rate']):.3f} | "
                    f"[{float(row['bootstrap_mean_ci_low']):.4f}, {float(row['bootstrap_mean_ci_high']):.4f}] |"
                )
        lines.append("")
    if external_scale:
        lines.append("### 外部基线优势区间")
        lines.append("")
        lines.append("这里的 `win external` 表示同一个 crop 上 SC-INR 同时优于 LIIF 和 LTE。")
        lines.append("注意：区域中位数仍可能被未同时赢两个模型的样本拉低；因此论文展示应绑定")
        lines.append("完整候选池统计、top/neutral/failure 案例，而不是只展示 top crop。")
        lines.append("")
        lines.append("| Scale | n | median vs LIIF | median vs LTE | win external | q75 min external Δ |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in external_scale:
            if row["split"] != "OOD":
                continue
            lines.append(
                f"| {row['scale']} | {row['n']} | {float(row['median_liif_delta']):.4f} | "
                f"{float(row['median_lte_delta']):.4f} | {float(row['win_rate_external']):.3f} | "
                f"{float(row['q75_min_external_delta']):.4f} |"
            )
        lines.append("")
    if external_pool:
        lines.append("### 论文定性图候选")
        lines.append("")
        lines.append("`external_advantage_region_pool.csv` 按外部基线优势排序，并保留")
        lines.append("top/neutral/failure。当前已人工筛选保留下来的主候选在")
        lines.append("`figures/external_advantage_crops/`；更适合正文展示的大上下文版本在")
        lines.append("`figures/external_advantage_context_zoom/`。若重新运行 `render_external`，")
        lines.append("脚本会重新生成带 `_clean.png` 后缀的纯 crop panel；当前目录内容以实际保留文件为准。")
        lines.append("")
        lines.append("| Type | Dataset/Image | Scale | Crop | bins | min Δ vs LIIF/LTE |")
        lines.append("| --- | --- | --- | --- | --- | ---: |")
        for row in external_pool:
            if row["representative_type"] != "external_top_positive":
                continue
            lines.append(
                f"| top | {row['dataset']}/{row['image']} | {row['scale']} | "
                f"{row['crop_id']}@({row['crop_y']},{row['crop_x']}) | "
                f"{row['texture_bin']}/{row['edge_bin']}/{row['highpass_bin']} | "
                f"{float(row['min_external_delta']):.3f} |"
            )
        lines.append("")
        lines.append("目前最适合正文的叙述是：在预注册候选池里，SC-INR 相对 LIIF/LTE 的")
        lines.append("明显视觉优势主要出现在 OOD 建筑/纹理/高频结构 crop，尤其是 Urban100 的")
        lines.append("x8/x16/x30 局部结构区域；但总体外部 win rate 仍不到 50%，所以这些是")
        lines.append("selected advantage regions，不是全局普遍优势。")
        lines.append("")
    if x4_pool:
        lines.append("### x4 sanity / 边界对照")
        lines.append("")
        lines.append("x4 属于当前训练/ID 尺度范围，不是本文主张的 OOD 优势区间。这里保留少量")
        lines.append("x4 neutral/sanity crop，用来说明低倍率下各方法差距通常很小，避免正文只展示")
        lines.append("x8-x30 时被误解为刻意隐藏低倍率结果。")
        lines.append("")
        lines.append("| Dataset/Image | Crop | bins | min Δ vs LIIF/LTE |")
        lines.append("| --- | --- | --- | ---: |")
        for row in x4_pool:
            lines.append(
                f"| {row['dataset']}/{row['image']} | {row['crop_id']}@({row['crop_y']},{row['crop_x']}) | "
                f"{row['texture_bin']}/{row['edge_bin']}/{row['highpass_bin']} | {float(row['min_external_delta']):.4f} |"
            )
        lines.append("")
    lines.extend([
        "## 产物",
        "",
        "- `per_image_per_scale_metrics.csv`：逐 seed / image / scale / model 的 PSNR-Y 等质量指标。",
        "- `paired_delta_per_image_scale.csv`：SC-INR 相对 LIIF/LTE 的逐样本 delta。",
        "- `paired_delta_*summary*.csv` 和 `paired_delta_bootstrap_ci.csv`：统计摘要和 cluster bootstrap CI。",
        "- `local_crop_descriptors.csv`：只由 GT 决定的 crop 坐标和 texture/edge/highpass 分层。",
        "- `local_crop_metrics.csv`：每个 crop 的模型质量指标。",
        "- `external_advantage_per_crop.csv`：每个 crop 上 SC-INR 相对 LIIF/LTE 的外部优势。",
        "- `external_advantage_region_pool.csv`：论文定性图候选，含 top/neutral/failure。",
        "- `figures/external_advantage_crops/`：用户筛选后保留的正文/appendix 候选 crop panel；目录内容以实际文件为准。",
        "- `figures/external_advantage_context_zoom/`：围绕已筛选外部优势图生成的更大上下文 + zoom panel，优先用于正文候选。",
        "- `external_advantage_x4_sanity_pool.csv` 和 `figures/external_advantage_x4_sanity/`：x4 边界对照，不作为主优势证据。",
        "- `local_crop_paired_deltas.csv`、`local_delta_summary_*`、`advantage_zone_*`、`advantage_margin_*` 仅在显式内部诊断模式下生成。",
        "",
        "## 使用边界",
        "",
        "- 该目录不能证明 strict scale equivariance、sinc 唯一因果或 exact RGB box integral。",
        "- 正文图只能说明 selected external-baseline advantage regions，必须和候选池统计一起解释。",
        "- x4 sanity 只用于展示 ID 低倍率边界，不能写成 SC-INR 在 x4 也有明显视觉优势。",
        "- 当前结果不支持“SC-INR 在所有局部区域普遍优于 LIIF/LTE”的强说法。",
        "- 消融比较后续单独讨论；本 README 的主展示不把 NoPhi/NoSinc 作为视觉主线。",
        "- 如果 win-rate 接近 50% 或 bootstrap CI 跨 0，应收缩为 modest average gain / diagnostic evidence。",
        "- same-LR consistency 不在本目录中作为 footprint correctness 主证据。",
    ])
    (out_dir / "README_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SC-INR per-image statistics and pre-registered local advantage regions.")
    parser.add_argument("--mode", choices=["all", "per_image", "local", "summarize", "render_external", "render_context", "render_x4"], default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "derived" / "diagnostics" / "sc_inr_advantage_2026-05-21")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--eval_bsize", type=int, default=50000)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--per_image_models", default="LIIF,LTE,SC-INR")
    parser.add_argument("--per_image_datasets", default="set5,set14,bsd100,urban100")
    parser.add_argument("--per_image_scales", default="2,3,4,6,8,12,16,24,30")
    parser.add_argument("--per_image_max_images", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260521)
    parser.add_argument("--local_models", default="LIIF,LTE,SC-INR,SC-INR-NoPhi,SC-INR-NoSinc")
    parser.add_argument("--local_datasets", default="bsd100,urban100")
    parser.add_argument("--local_scales", default="4,8,16,30")
    parser.add_argument("--local_max_images", type=int, default=30)
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--crop_stride", type=int, default=96)
    parser.add_argument("--max_crops_per_image_scale", type=int, default=32)
    parser.add_argument("--render_top_k", type=int, default=4)
    parser.add_argument("--render_neutral_k", type=int, default=2)
    parser.add_argument("--render_failure_k", type=int, default=2)
    parser.add_argument("--render_x4_k", type=int, default=4)
    parser.add_argument("--context_factor", type=int, default=3)
    parser.add_argument("--make_internal_plots", action="store_true", help="生成内部统计热图/柱状图；默认关闭，论文展示优先用 external advantage crop。")
    parser.add_argument("--render_internal_examples", action="store_true", help="生成包含 NoPhi/NoSinc 的内部代表图；默认关闭。")
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    run_config = vars(args).copy()
    run_config["device_resolved"] = str(device)
    (out_dir / "run_config.json").write_text(json.dumps(json_safe(run_config), indent=2, ensure_ascii=False), encoding="utf-8")

    per_image_rows: List[Dict[str, object]] | None = None
    local_rows: List[Dict[str, object]] | None = None
    if args.mode in {"all", "per_image"}:
        per_image_rows = run_per_image(args, device, out_dir)
        summarize_per_image(per_image_rows, out_dir, args.n_boot, args.bootstrap_seed, make_internal_plots=args.make_internal_plots)
        regression_against_canonical(out_dir)
    elif args.mode == "summarize":
        rows = [dict(r) for r in read_csv(out_dir / "per_image_per_scale_metrics.csv")]
        if rows:
            summarize_per_image(rows, out_dir, args.n_boot, args.bootstrap_seed, make_internal_plots=args.make_internal_plots)
            regression_against_canonical(out_dir)

    if args.mode in {"all", "local"}:
        local_rows = run_local_crops(args, device, out_dir)
    elif args.mode == "summarize":
        rows = [dict(r) for r in read_csv(out_dir / "local_crop_metrics.csv")]
        if rows:
            summarize_local_crops(rows, out_dir, make_internal_plots=args.make_internal_plots)
    elif args.mode == "render_external":
        render_external_advantage_examples(args, device, out_dir)
    elif args.mode == "render_context":
        render_context_zoom_examples(args, device, out_dir)
    elif args.mode == "render_x4":
        render_x4_sanity_examples(args, device, out_dir)

    write_readme(out_dir, args, per_image_rows, local_rows)
    print(f"SC-INR advantage diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
