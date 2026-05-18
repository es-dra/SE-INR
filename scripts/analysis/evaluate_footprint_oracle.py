#!/usr/bin/env python3
"""Footprint-oracle diagnostics for decoder-side cell responses.

This diagnostic keeps the LR input and query coordinates fixed at an x4 target
grid, changes only the decoder cell size, and compares the resulting prediction
against a box-averaged HR oracle at the same query locations. It is designed to
separate footprint tracking from same-LR self-consistency: a cell-insensitive
decoder can be very self-consistent while failing to follow the oracle target as
the footprint changes.

The oracle is a discrete, piecewise-constant proxy built from the available HR
image. It is not a claim that the finite HR image is the true continuous scene.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import utils
from scripts.analysis.evaluate_seed1_aux_metrics import (
    DATASETS,
    ensure_dir,
    list_images,
    load_model,
    make_lr_hr,
    psnr_from_mse,
    rgb_to_y,
    shave_valid,
)
from scripts.analysis.model_registry import MODEL_PATHS, STYLE


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]], keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)

    out: List[Dict[str, object]] = []
    for group_key, items in sorted(groups.items()):
        rec: Dict[str, object] = {k: v for k, v in zip(keys, group_key)}
        rec["n"] = len(items)
        rec["weak_oracle_frac"] = float(np.mean([float(bool(r["weak_oracle"])) for r in items]))
        for metric in metrics:
            vals = []
            for row in items:
                value = row.get(metric)
                if value in (None, ""):
                    continue
                value = float(value)
                if math.isfinite(value):
                    vals.append(value)
            rec[metric] = float(np.mean(vals)) if vals else float("nan")
        out.append(rec)
    return out


def footprint_weights(multiplier: float, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return 1D piecewise-constant box weights centered on a pixel center.

    For multiplier 2 this gives [0.25, 0.5, 0.25], i.e. the exact overlap of a
    width-2 box with unit-width pixels under a piecewise-constant HR proxy.
    """

    if multiplier <= 0:
        raise ValueError("cell multiplier must be positive")
    radius = float(multiplier) / 2.0
    lo = math.floor(-radius - 0.5) - 1
    hi = math.ceil(radius + 0.5) + 1
    weights = []
    for offset in range(lo, hi + 1):
        pix_lo = offset - 0.5
        pix_hi = offset + 0.5
        overlap = max(0.0, min(pix_hi, radius) - max(pix_lo, -radius))
        if overlap > 1e-12:
            weights.append(overlap / float(multiplier))
    if not weights:
        raise RuntimeError(f"empty footprint weights for multiplier {multiplier}")
    w = torch.tensor(weights, device=device, dtype=dtype)
    return w / w.sum()


def box_oracle(img: torch.Tensor, multiplier: float) -> torch.Tensor:
    """Apply a separable centered box average to an HR image tensor."""

    if abs(multiplier - 1.0) < 1e-12:
        return img
    w = footprint_weights(multiplier, img.device, img.dtype)
    k = int(w.numel())
    pad_left = k // 2
    pad_right = k - 1 - pad_left

    channels = img.shape[1]
    weight_x = w.view(1, 1, 1, k).repeat(channels, 1, 1, 1)
    weight_y = w.view(1, 1, k, 1).repeat(channels, 1, 1, 1)
    out = F.pad(img, (pad_left, pad_right, 0, 0), mode="replicate")
    out = F.conv2d(out, weight_x, groups=channels)
    out = F.pad(out, (0, 0, pad_left, pad_right), mode="replicate")
    out = F.conv2d(out, weight_y, groups=channels)
    return out


def make_coord_cell(h: int, w: int, multiplier: float, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    coord = utils.make_coord([h, w]).unsqueeze(0).to(device)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2.0 * float(multiplier) / h
    cell[:, :, 1] *= 2.0 * float(multiplier) / w
    return coord.contiguous(), cell.contiguous()


def predict_with_cell(
    model,
    lr: torch.Tensor,
    h: int,
    w: int,
    multiplier: float,
    device: torch.device | str,
    bsize: int,
    feat_ready: bool,
) -> torch.Tensor:
    inp_sub = torch.tensor([0.5], device=device).view(1, -1, 1, 1)
    inp_div = torch.tensor([0.5], device=device).view(1, -1, 1, 1)
    gt_sub = torch.tensor([0.5], device=device).view(1, 1, -1)
    gt_div = torch.tensor([0.5], device=device).view(1, 1, -1)
    inp = (lr - inp_sub) / inp_div
    coord, cell = make_coord_cell(h, w, multiplier, device)

    with torch.no_grad():
        if not feat_ready:
            model.gen_feat(inp)
        preds = []
        for ql in range(0, coord.shape[1], bsize):
            qr = min(ql + bsize, coord.shape[1])
            preds.append(model.query_rgb(coord[:, ql:qr], cell[:, ql:qr]))
        pred = torch.cat(preds, dim=1)

    pred = (pred * gt_div + gt_sub).clamp(0, 1)
    return pred.view(1, h, w, 3).permute(0, 3, 1, 2).contiguous()


def metric_dict(
    pred: torch.Tensor,
    pred_base: torch.Tensor,
    oracle: torch.Tensor,
    oracle_base: torch.Tensor,
    base_scale: int,
    multiplier: float,
    weak_threshold: float,
) -> Dict[str, object]:
    shave = max(int(base_scale), int(math.ceil(float(multiplier) / 2.0)) + 1)
    pred_y = shave_valid(rgb_to_y(pred), shave)
    base_y = shave_valid(rgb_to_y(pred_base), shave)
    oracle_y = shave_valid(rgb_to_y(oracle), shave)
    oracle_base_y = shave_valid(rgb_to_y(oracle_base), shave)

    oracle_diff = pred_y - oracle_y
    oracle_mse = float(oracle_diff.square().mean().item())
    pred_delta = pred_y - base_y
    oracle_delta = oracle_y - oracle_base_y
    tracking = pred_delta - oracle_delta
    cell_sens = pred_delta
    oracle_change = oracle_delta

    delta_tracking_rmse = float(torch.sqrt(tracking.square().mean()).item())
    cell_sensitivity_rmse = float(torch.sqrt(cell_sens.square().mean()).item())
    oracle_change_rmse = float(torch.sqrt(oracle_change.square().mean()).item())
    return {
        "oracle_psnr_y": psnr_from_mse(oracle_mse),
        "oracle_rmse_y": math.sqrt(oracle_mse),
        "delta_tracking_rmse_y": delta_tracking_rmse,
        "cell_sensitivity_rmse_y": cell_sensitivity_rmse,
        "oracle_change_rmse_y": oracle_change_rmse,
        "weak_oracle": oracle_change_rmse < weak_threshold and abs(float(multiplier) - 1.0) > 1e-12,
        "valid_shave": shave,
    }


def plot_summary(summary_path: Path, fig_dir: Path) -> None:
    if not summary_path.exists():
        return
    import pandas as pd

    df = pd.read_csv(summary_path)
    if df.empty:
        return
    ensure_dir(fig_dir)
    df = df[df["cell_multiplier"].astype(float) > 1.0].copy()
    if df.empty:
        return
    df["cell_multiplier_num"] = df["cell_multiplier"].astype(float)
    for metric, ylabel, filename in [
        ("delta_tracking_rmse_y", "Delta tracking RMSE-Y (lower is better)", "delta_tracking_rmse_y.png"),
        ("oracle_rmse_y", "Oracle RMSE-Y (lower is better)", "oracle_rmse_y.png"),
        ("cell_sensitivity_rmse_y", "Cell sensitivity RMSE-Y", "cell_sensitivity_rmse_y.png"),
    ]:
        datasets = sorted(df["dataset"].unique())
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, dataset in zip(axes, datasets):
            sub_ds = df[df["dataset"] == dataset]
            for model_name, sub in sub_ds.groupby("model"):
                sub = sub.sort_values("cell_multiplier_num")
                ax.plot(
                    sub["cell_multiplier_num"],
                    sub[metric],
                    marker="o",
                    label=model_name,
                    color=STYLE.get(model_name),
                )
            ax.set_title(dataset)
            ax.set_xlabel("Cell multiplier")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[-1].legend(handles, labels, fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=220)
        plt.close(fig)


def write_readme(out_dir: Path, args: argparse.Namespace, conclusion: Dict[str, object]) -> None:
    text = f"""# Footprint Oracle Diagnostics 2026-05-15

本目录评估 decoder cell response 是否跟随真实 footprint 观测，而不是只比较模型输出之间的
same-LR consistency。

## 协议

- 固定 LR：HR 图像 bicubic downsample 到 x{args.base_scale} LR。
- 固定 query：x{args.base_scale} HR grid。
- 只改变 decoder 输入的 cell multiplier：`{args.cell_multipliers}`。
- oracle target：在同一 HR grid 上，对 HR 图像做 piecewise-constant box average。
- 模型：`{args.models}`。
- 数据：`{args.datasets}` sorted 前 `{args.max_images}` 张。

## 指标

- `oracle_psnr_y` / `oracle_rmse_y`：模型输出 vs box-averaged oracle target。
- `delta_tracking_rmse_y`：模型 cell-change 与 oracle cell-change 的差异，越低越好。
- `cell_sensitivity_rmse_y`：模型相对 native cell 的输出变化幅度。
- `oracle_change_rmse_y`：真实 oracle 相对 native footprint 的变化幅度。
- `weak_oracle`：当 oracle 本身变化低于 `{args.weak_oracle_threshold}` 时标记，不能用于强结论。

## 当前 gate 摘要

- gate_status: `{conclusion.get("gate_status", "unknown")}`
- best_vs_nosinc_delta_tracking: `{conclusion.get("best_vs_nosinc_delta_tracking", "n/a")}`
- best_vs_nosinc_oracle_rmse: `{conclusion.get("best_vs_nosinc_oracle_rmse", "n/a")}`

## 解释边界

- 这是 HR piecewise-constant box oracle，不是真实连续场景的精确积分。
- 该诊断可检验 cell response 是否跟随 footprint target，但不能单独证明 sinc 是唯一因果。
- same-LR consistency 仍然是 diagnostic-only；cell-insensitive 模型可能 self-consistency 很高。
- 若 `SC-INR-NoSinc` 或其他负控更好，应收缩 claim，而不是追加相邻指标寻找正结果。

## 文件

- `footprint_oracle_metrics.csv`：逐图、逐模型、逐 cell multiplier 指标。
- `footprint_oracle_summary.csv`：按 dataset/model/cell multiplier 汇总。
- `footprint_oracle_overall.csv`：按 model/cell multiplier 汇总。
- `figures/`：摘要图。
- `run_config.json`：运行参数。
"""
    (out_dir / "README_zh.md").write_text(text)


def gate_conclusion(overall_rows: List[Dict[str, object]]) -> Dict[str, object]:
    rows = [r for r in overall_rows if str(r.get("cell_multiplier")) != "1"]
    def mean_metric(model: str, metric: str) -> float:
        vals = [float(r[metric]) for r in rows if r["model"] == model and math.isfinite(float(r[metric]))]
        return float(np.mean(vals)) if vals else float("nan")

    sc_track = mean_metric("SC-INR", "delta_tracking_rmse_y")
    nosinc_track = mean_metric("SC-INR-NoSinc", "delta_tracking_rmse_y")
    sc_oracle = mean_metric("SC-INR", "oracle_rmse_y")
    nosinc_oracle = mean_metric("SC-INR-NoSinc", "oracle_rmse_y")
    track_win = math.isfinite(sc_track) and math.isfinite(nosinc_track) and sc_track < nosinc_track
    oracle_win = math.isfinite(sc_oracle) and math.isfinite(nosinc_oracle) and sc_oracle < nosinc_oracle
    if track_win or oracle_win:
        status = "supports_sc_inr_over_nosinc_on_at_least_one_primary_metric"
    else:
        status = "mixed_or_negative_do_not_strengthen_sinc_claim"
    return {
        "gate_status": status,
        "best_vs_nosinc_delta_tracking": sc_track - nosinc_track if math.isfinite(sc_track) and math.isfinite(nosinc_track) else "nan",
        "best_vs_nosinc_oracle_rmse": sc_oracle - nosinc_oracle if math.isfinite(sc_oracle) and math.isfinite(nosinc_oracle) else "nan",
    }


def run_analysis(args: argparse.Namespace) -> None:
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model_names = parse_csv_list(args.models)
    missing = [m for m in model_names if m not in MODEL_PATHS or not MODEL_PATHS[m].exists()]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints: {missing}")
    multipliers = parse_float_list(args.cell_multipliers)
    if 1.0 not in multipliers:
        raise ValueError("--cell_multipliers must include 1 for native-cell reference")

    rows: List[Dict[str, object]] = []
    loaded = {name: load_model(name, device) for name in model_names}
    for dataset in parse_csv_list(args.datasets):
        for img_path in list_images(dataset, args.max_images):
            lr, hr = make_lr_hr(img_path, args.base_scale, device)
            h, w = hr.shape[-2:]
            oracles = {m: box_oracle(hr, m).clamp(0, 1) for m in multipliers}
            oracle_base = oracles[1.0]
            for model_name, model in loaded.items():
                with torch.no_grad():
                    inp = (lr - 0.5) / 0.5
                    model.gen_feat(inp)
                    preds = {
                        m: predict_with_cell(
                            model,
                            lr,
                            h,
                            w,
                            m,
                            device,
                            args.eval_bsize,
                            feat_ready=True,
                        )
                        for m in multipliers
                    }
                pred_base = preds[1.0]
                for multiplier in multipliers:
                    rec: Dict[str, object] = {
                        "model": model_name,
                        "dataset": dataset,
                        "image": img_path.name,
                        "base_scale": f"x{args.base_scale}",
                        "cell_multiplier": f"{multiplier:g}",
                    }
                    rec.update(
                        metric_dict(
                            preds[multiplier],
                            pred_base,
                            oracles[multiplier],
                            oracle_base,
                            args.base_scale,
                            multiplier,
                            args.weak_oracle_threshold,
                        )
                    )
                    rows.append(rec)
                del preds
                torch.cuda.empty_cache()
            del lr, hr, oracles
            torch.cuda.empty_cache()

    metrics = [
        "oracle_psnr_y",
        "oracle_rmse_y",
        "delta_tracking_rmse_y",
        "cell_sensitivity_rmse_y",
        "oracle_change_rmse_y",
    ]
    write_csv(out_dir / "footprint_oracle_metrics.csv", rows)
    summary = summarize(rows, ["dataset", "model", "cell_multiplier"], metrics)
    overall = summarize(rows, ["model", "cell_multiplier"], metrics)
    write_csv(out_dir / "footprint_oracle_summary.csv", summary)
    write_csv(out_dir / "footprint_oracle_overall.csv", overall)
    plot_summary(out_dir / "footprint_oracle_summary.csv", out_dir / "figures")
    conclusion = gate_conclusion(overall)
    (out_dir / "gate_conclusion.json").write_text(json.dumps(conclusion, indent=2, default=str))
    write_readme(out_dir, args, conclusion)
    print(f"Footprint-oracle diagnostics written to {out_dir}")
    print(json.dumps(conclusion, indent=2, default=str))


def run_self_test() -> None:
    w2 = footprint_weights(2.0, "cpu")
    expected_w2 = torch.tensor([0.25, 0.5, 0.25])
    if not torch.allclose(w2, expected_w2, atol=1e-7):
        raise AssertionError(f"unexpected width-2 weights: {w2}")
    w4 = footprint_weights(4.0, "cpu")
    expected_w4 = torch.tensor([0.125, 0.25, 0.25, 0.25, 0.125])
    if not torch.allclose(w4, expected_w4, atol=1e-7):
        raise AssertionError(f"unexpected width-4 weights: {w4}")

    h, w = 96, 96
    x = torch.arange(w).float().view(1, 1, 1, w)
    omega = 0.1
    signal = torch.cos(math.pi * omega * x).repeat(1, 1, h, 1)
    amp1 = box_oracle(signal, 1.0)[..., 16:-16, 16:-16].std()
    amp2 = box_oracle(signal, 2.0)[..., 16:-16, 16:-16].std()
    amp4 = box_oracle(signal, 4.0)[..., 16:-16, 16:-16].std()
    ratio2 = float(amp2 / amp1)
    ratio4 = float(amp4 / amp1)
    sinc2 = float(torch.sinc(torch.tensor(omega * 2.0 / 2.0)))
    sinc4 = float(torch.sinc(torch.tensor(omega * 4.0 / 2.0)))
    if not (ratio4 < ratio2 < 1.0):
        raise AssertionError(f"expected stronger attenuation for larger footprints, got {ratio2}, {ratio4}")
    if abs(ratio2 - sinc2) > 0.03 or abs(ratio4 - sinc4) > 0.03:
        raise AssertionError(
            f"discrete oracle should approximate continuous sinc for low omega; "
            f"got ratios {ratio2:.4f}/{ratio4:.4f}, sinc {sinc2:.4f}/{sinc4:.4f}"
        )

    const = torch.ones(1, 3, 12, 13)
    for multiplier in (1.0, 2.0, 4.0):
        out = box_oracle(const, multiplier)
        if not torch.allclose(out, const, atol=1e-7):
            raise AssertionError("box oracle must preserve constants")
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate footprint-oracle cell response diagnostics.")
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--models", default="LTE,LTE-NoCellPhase,SC-INR-NoPhi,SC-INR,SC-INR-NoSinc")
    parser.add_argument("--datasets", default="bsd100,urban100")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--base_scale", type=int, default=4)
    parser.add_argument("--cell_multipliers", default="1,2,4")
    parser.add_argument("--weak_oracle_threshold", type=float, default=0.002)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_bsize", type=int, default=50000)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "derived" / "diagnostics" / "footprint_oracle_2026-05-15",
    )
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    os.chdir(ROOT)
    run_analysis(args)


if __name__ == "__main__":
    main()
