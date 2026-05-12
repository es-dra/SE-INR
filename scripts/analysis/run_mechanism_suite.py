#!/usr/bin/env python3
"""Paper-facing mechanism diagnostics for SC-INR-style decoders.

This CLI complements the existing benchmark and sampling-response scripts. It
keeps the six mechanism checks in one artifact tree:

6.2 scale gain curve;
6.5 OOD cell extrapolation diagnostic;
6.6 cell intervention patches.

6.1 cell response curves and 6.4 frequency/response distributions are produced
by scripts/analysis/analyze_sampling_response.py. 6.3 sampling consistency is
produced by scripts/analysis/evaluate_seed1_aux_metrics.py. This script writes a
README that indexes all six outputs when they share the same suite directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import utils
from scripts.analysis.model_registry import canonicalize_result_key
from scripts.analysis.analyze_sampling_response import (
    DATASETS,
    STYLE,
    effective_omega_components,
    ensure_dir,
    list_images,
    load_model,
    make_lr_hr,
    parse_csv_list,
    parse_scales,
    rgb_to_y,
    scale_label,
    summarize_tensor,
)


ID_SCALES = [2, 3, 4]
OOD_SCALES = [6, 8, 12, 16, 24, 30]
BENCHMARKS = ["Set5", "Set14", "BSD100", "Urban100"]


def normalize_model_names(data: Dict[str, object], source_path: Path | None = None) -> Dict[str, object]:
    raw_keys = set(data.keys())
    normalized: Dict[str, object] = {}
    for model, value in data.items():
        normalized[canonicalize_result_key(model, source_path, raw_keys)] = value
    return normalized


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_merged_benchmark(paths: Sequence[Path]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r") as f:
            data = normalize_model_names(json.load(f), path)
        merged.update(data)
    return merged


def scale_means(model_data: Dict[str, object], scale: int) -> List[float]:
    vals: List[float] = []
    for dataset in BENCHMARKS:
        v = model_data.get(dataset, {}).get(f"x{scale}")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return vals


def run_scale_gain(args: SimpleNamespace, out_dir: Path) -> None:
    bench = load_merged_benchmark([Path(p) for p in parse_csv_list(args.benchmark_jsons)])
    pairs = []
    for chunk in args.scale_gain_pairs.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        model, baseline = [x.strip() for x in chunk.split(",")]
        pairs.append((model, baseline))

    rows: List[Dict[str, object]] = []
    for model, baseline in pairs:
        if model not in bench or baseline not in bench:
            continue
        for scale in ID_SCALES + OOD_SCALES:
            model_vals = scale_means(bench[model], scale)
            baseline_vals = scale_means(bench[baseline], scale)
            paired = list(zip(model_vals, baseline_vals))
            if not paired:
                continue
            deltas = [a - b for a, b in paired]
            split = "ID" if scale in ID_SCALES else "OOD"
            rows.append(
                {
                    "experiment": "6.2_scale_gain_curve",
                    "model": model,
                    "baseline": baseline,
                    "scale": f"x{scale}",
                    "split": split,
                    "model_psnr_mean": float(np.mean(model_vals)),
                    "baseline_psnr_mean": float(np.mean(baseline_vals)),
                    "delta_psnr_mean": float(np.mean(deltas)),
                    "n_datasets": len(paired),
                }
            )

    write_csv(out_dir / "scale_gain_curve.csv", rows)
    if not rows:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for key, sub_rows in group_rows(rows, ["model", "baseline"]).items():
        model, baseline = key
        sub_rows = sorted(sub_rows, key=lambda r: int(str(r["scale"]).replace("x", "")))
        xs = [int(str(r["scale"]).replace("x", "")) for r in sub_rows]
        ys = [float(r["delta_psnr_mean"]) for r in sub_rows]
        ax.plot(xs, ys, marker="o", label=f"{model} - {baseline}", color=STYLE.get(model, None))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(1.5, 4.5, color="#DDDDDD", alpha=0.22, label="train scale range")
    ax.set_xlabel("Scale")
    ax.set_ylabel("PSNR-Y gain (dB)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "scale_gain_curve.png", dpi=220)
    fig.savefig(fig_dir / "scale_gain_curve.pdf")
    plt.close(fig)


def group_rows(rows: Iterable[Dict[str, object]], keys: Sequence[str]) -> Dict[tuple, List[Dict[str, object]]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return groups


def make_query_coords(size: Sequence[int], max_queries: int, device: torch.device) -> torch.Tensor:
    coord = utils.make_coord(size).unsqueeze(0).to(device)
    if coord.shape[1] > max_queries:
        idx = torch.linspace(0, coord.shape[1] - 1, steps=max_queries, device=device).long()
        coord = coord[:, idx, :]
    return coord.contiguous()


def tensor_stats(x: torch.Tensor, prefix: str = "signal") -> Dict[str, float]:
    stats = summarize_tensor(x, prefix)
    vals = x.detach().float().flatten()
    stats[f"{prefix}_rms"] = float(torch.sqrt(vals.square().mean()).item()) if vals.numel() else float("nan")
    return stats


def lte_cell_signal(model, coord: torch.Tensor, rel_cell: torch.Tensor) -> torch.Tensor | None:
    bs, q = coord.shape[:2]
    if hasattr(model, "phase"):
        if hasattr(model, "tranNum") and model.__class__.__module__.endswith("lte_eq"):
            rel_cellx = rel_cell[:, :, 0].unsqueeze(2).repeat([1, 1, model.tranNum])
            rel_celly = rel_cell[:, :, 1].unsqueeze(2).repeat([1, 1, model.tranNum])
            rel_cell_eq = torch.cat([rel_cellx, rel_celly], dim=-1)
            return model.phase(rel_cell_eq.reshape((bs * q, -1))).reshape(bs, q, -1)
        return model.phase(rel_cell.reshape((bs * q, -1))).reshape(bs, q, -1)
    if hasattr(model, "phase_map"):
        q_phase = F.grid_sample(
            model.phase_map,
            coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        return q_phase
    return None


def sc_response_signal(model, scale: float) -> torch.Tensor | None:
    effective_omega = effective_omega_components(model)
    if effective_omega is None:
        return None
    omega_x = effective_omega[..., 0]
    omega_y = effective_omega[..., 1]
    rel_cell = 2.0 / float(scale)
    response = torch.sinc(omega_x * rel_cell / 2) * torch.sinc(omega_y * rel_cell / 2)
    if not bool(getattr(model, "use_sinc_response", True)):
        response = torch.ones_like(response)
    return response


def run_cell_extrapolation(args: SimpleNamespace, device: torch.device, out_dir: Path) -> None:
    models = parse_csv_list(args.cell_signal_models)
    scales = parse_scales(args.cell_signal_scales)
    dataset = args.cell_signal_dataset
    images = list_images(dataset, args.cell_signal_max_images)
    if not images:
        raise FileNotFoundError(f"No images for {dataset}")
    image_path = images[0]
    lr, hr = make_lr_hr(image_path, args.cell_signal_lr_scale, device)
    coord = make_query_coords(hr.shape[-2:], args.max_queries, device)

    rows: List[Dict[str, object]] = []
    for model_name in models:
        model = load_model(model_name, device)
        with torch.no_grad():
            model.gen_feat((lr - 0.5) / 0.5)
            ref_flat = None
            for scale in scales:
                cell = torch.ones_like(coord)
                cell[:, :, 0] *= 2 / (lr.shape[-2] * float(scale))
                cell[:, :, 1] *= 2 / (lr.shape[-1] * float(scale))
                rel_cell = cell.clone()
                feat_h = getattr(model, "feat", getattr(model, "coeff", None)).shape[-2]
                feat_w = getattr(model, "feat", getattr(model, "coeff", None)).shape[-1]
                rel_cell[:, :, 0] *= feat_h
                rel_cell[:, :, 1] *= feat_w

                signal = sc_response_signal(model, scale)
                signal_type = "analytic_sinc_response"
                if signal is None:
                    signal = lte_cell_signal(model, coord, rel_cell)
                    signal_type = "learned_or_feature_phase"
                if signal is None:
                    signal = torch.zeros(coord.shape[0], coord.shape[1], 1, device=device)
                    signal_type = "no_cell_signal"
                flat = signal.detach().float().flatten()
                if ref_flat is None:
                    ref_flat = flat
                n = min(flat.numel(), ref_flat.numel())
                delta = torch.sqrt((flat[:n] - ref_flat[:n]).square().mean()) if n else torch.tensor(float("nan"))

                row: Dict[str, object] = {
                    "experiment": "6.5_ood_cell_extrapolation",
                    "model": model_name,
                    "signal_type": signal_type,
                    "dataset": dataset,
                    "image": image_path.name,
                    "scale": scale_label(scale),
                    "inside_train_scale_range": bool(1.0 <= float(scale) <= 4.0),
                    "delta_rms_vs_first_cell": float(delta),
                }
                row.update(tensor_stats(signal, "signal"))
                rows.append(row)
        del model
        torch.cuda.empty_cache()
    del lr, hr, coord
    torch.cuda.empty_cache()

    write_csv(out_dir / "cell_extrapolation_diagnostic.csv", rows)
    if not rows:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for model_name, sub_rows in group_rows(rows, ["model"]).items():
        sub_rows = sorted(sub_rows, key=lambda r: float(str(r["scale"]).replace("x", "")))
        xs = [float(str(r["scale"]).replace("x", "")) for r in sub_rows]
        ys = [float(r["signal_rms"]) for r in sub_rows]
        ax.plot(xs, ys, marker="o", label=model_name[0], color=STYLE.get(model_name[0], None))
    ax.axvspan(1.0, 4.0, color="#DDDDDD", alpha=0.22, label="train scale range")
    ax.set_xlabel("Cell-equivalent output scale")
    ax.set_ylabel("Signal RMS")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "cell_extrapolation_signal_rms.png", dpi=220)
    fig.savefig(fig_dir / "cell_extrapolation_signal_rms.pdf")
    plt.close(fig)


def patch_coords(target_h: int, target_w: int, crop_size: int, device: torch.device) -> tuple[torch.Tensor, int, int, int]:
    crop = min(crop_size, target_h, target_w)
    top = max(0, (target_h - crop) // 2)
    left = max(0, (target_w - crop) // 2)
    full = utils.make_coord([target_h, target_w]).view(target_h, target_w, 2)
    coord = full[top : top + crop, left : left + crop].reshape(1, -1, 2).to(device)
    return coord.contiguous(), top, left, crop


def query_patch(
    model,
    lr: torch.Tensor,
    coord: torch.Tensor,
    target_h: int,
    target_w: int,
    cell_multiplier: float,
    crop: int,
    bsize: int,
) -> torch.Tensor:
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= (2 / target_h) * float(cell_multiplier)
    cell[:, :, 1] *= (2 / target_w) * float(cell_multiplier)
    with torch.no_grad():
        model.gen_feat((lr - 0.5) / 0.5)
        preds = []
        for ql in range(0, coord.shape[1], bsize):
            qr = min(ql + bsize, coord.shape[1])
            preds.append(model.query_rgb(coord[:, ql:qr], cell[:, ql:qr]))
        pred = torch.cat(preds, dim=1)
    pred = (pred * 0.5 + 0.5).clamp(0, 1)
    return pred.view(1, crop, crop, 3)


def run_cell_intervention(args: SimpleNamespace, device: torch.device, out_dir: Path) -> None:
    models = parse_csv_list(args.intervention_models)
    multipliers = [float(x) for x in parse_csv_list(args.cell_multipliers)]
    dataset = args.intervention_dataset
    image_path = DATASETS[dataset] / args.intervention_image
    if not image_path.exists():
        image_path = list_images(dataset, 1)[0]
    lr, _ = make_lr_hr(image_path, args.intervention_lr_scale, device)
    target_h = int(round(lr.shape[-2] * float(args.intervention_scale)))
    target_w = int(round(lr.shape[-1] * float(args.intervention_scale)))
    coord, top, left, crop = patch_coords(target_h, target_w, args.crop_size, device)

    rows: List[Dict[str, object]] = []
    fig_dir = out_dir / "figures" / "cell_intervention"
    ensure_dir(fig_dir)
    for model_name in models:
        model = load_model(model_name, device)
        patches: Dict[float, torch.Tensor] = {}
        ref = None
        for multiplier in multipliers:
            patch = query_patch(model, lr, coord, target_h, target_w, multiplier, crop, args.eval_bsize)
            patches[multiplier] = patch.detach().cpu()
            if abs(multiplier - 1.0) < 1e-9:
                ref = patch
        if ref is None:
            ref = patches[multipliers[0]].to(device)
        ref_y = rgb_to_y(ref.to(device).view(1, crop * crop, 3))
        for multiplier, patch_cpu in patches.items():
            patch = patch_cpu.to(device)
            patch_y = rgb_to_y(patch.view(1, crop * crop, 3))
            rmse = torch.sqrt((patch_y - ref_y).square().mean())
            rows.append(
                {
                    "experiment": "6.6_cell_intervention",
                    "model": model_name,
                    "dataset": dataset,
                    "image": image_path.name,
                    "scale": scale_label(float(args.intervention_scale)),
                    "cell_multiplier": multiplier,
                    "crop_top": top,
                    "crop_left": left,
                    "crop_size": crop,
                    "rmse_y_vs_multiplier_1": float(rmse),
                    "mean_y": float(patch_y.mean()),
                    "std_y": float(patch_y.std(unbiased=False)),
                }
            )

        fig, axes = plt.subplots(1, len(multipliers), figsize=(2.2 * len(multipliers), 2.4))
        if len(multipliers) == 1:
            axes = [axes]
        for ax, multiplier in zip(axes, multipliers):
            ax.imshow(patches[multiplier].squeeze(0).numpy())
            ax.set_title(f"cell x{multiplier:g}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{model_name} {dataset}/{image_path.name} x{args.intervention_scale:g}", fontsize=10)
        fig.tight_layout()
        safe_model = model_name.replace("/", "_").replace("+", "plus")
        fig.savefig(fig_dir / f"{safe_model}_{dataset}_{image_path.stem}_x{args.intervention_scale:g}.png", dpi=220)
        plt.close(fig)
        del model
        torch.cuda.empty_cache()
    del lr, coord
    torch.cuda.empty_cache()
    write_csv(out_dir / "cell_intervention_summary.csv", rows)


def write_readme(out_dir: Path, args: SimpleNamespace) -> None:
    text = f"""# Mechanism Suite 2026-05-12

本目录汇总 SC-INR/SC-INR-EQ 相关机制诊断。当前状态是
`diagnostic/exploratory`，不能直接升级为主论文 claim。

## 包含的检查

- 6.1 Cell response curve：`sampling_response/cell_response_curve.csv` 和 figures。
- 6.2 Scale gain curve：`scale_gain_curve.csv`。
- 6.3 Sampling consistency metric：`sampling_consistency/consistency_summary.csv`。
- 6.4 Frequency/response visualization：`sampling_response/response_distribution_summary.csv` 和 figures。
- 6.5 OOD cell extrapolation diagnostic：`cell_extrapolation_diagnostic.csv`。
- 6.6 Cell intervention experiment：`cell_intervention_summary.csv` 和 `figures/cell_intervention/`。

## 使用边界

这些诊断可以支持 mechanism discussion，但不能单独证明 strict scale
equivariance 或 whole-network sampling correctness。Same-LR consistency 必须和
fidelity、NoSinc 负控、response/cell intervention 一起解释。

## 主参数

```json
{json.dumps(vars(args), indent=2, default=str)}
```
"""
    (out_dir / "README_zh.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "derived" / "diagnostics" / "mechanism_suite_2026-05-12")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--benchmark_jsons",
        default="artifacts/raw_results/seed1/benchmark_signed_phiz.json,artifacts/raw_results/seed1/benchmark_sc_inr_eq.json",
    )
    parser.add_argument("--scale_gain_pairs", default="SC-INR,LTE;SC-INR-EQ,LTE-EQ")
    parser.add_argument("--cell_signal_models", default="LTE,LTE-NoCellPhase,LTE-PhaseZ,SC-INR,SC-INR-EQ")
    parser.add_argument("--cell_signal_dataset", default="urban100")
    parser.add_argument("--cell_signal_max_images", type=int, default=1)
    parser.add_argument("--cell_signal_lr_scale", type=int, default=4)
    parser.add_argument("--cell_signal_scales", default="1,2,3,4,6,8,12,16,24,30")
    parser.add_argument("--intervention_models", default="LTE,LTE-NoCellPhase,LTE-PhaseZ,SC-INR,SC-INR-EQ")
    parser.add_argument("--intervention_dataset", default="urban100")
    parser.add_argument("--intervention_image", default="img_004.png")
    parser.add_argument("--intervention_lr_scale", type=int, default=4)
    parser.add_argument("--intervention_scale", type=float, default=16)
    parser.add_argument("--cell_multipliers", default="0.5,1,2,4")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--max_queries", type=int, default=1024)
    parser.add_argument("--eval_bsize", type=int, default=50000)
    parser.add_argument(
        "--overwrite_readme",
        action="store_true",
        help="Overwrite README_zh.md. By default, keep an existing reviewed README intact.",
    )
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    run_scale_gain(args, out_dir)
    run_cell_extrapolation(args, device, out_dir)
    run_cell_intervention(args, device, out_dir)
    readme_path = out_dir / "README_zh.md"
    if args.overwrite_readme or not readme_path.exists():
        write_readme(out_dir, args)
    print(f"Mechanism suite diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
