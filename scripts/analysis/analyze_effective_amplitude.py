#!/usr/bin/env python3
"""Feature-level effective amplitude diagnostics for SC-INR decoders.

This diagnostic extracts the quantity used by the SC-INR decoder before the
MLP: coefficient pairs from z, analytic response W(omega, c), and the effective
coefficient-pair magnitude after multiplying by W. It does not claim that the
final RGB output is an exact Fourier amplitude, because the feature vector is
processed by an MLP and optionally merged by local ensemble.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from scripts.analysis.analyze_sampling_response import (
    DATASETS,
    STYLE,
    ensure_dir,
    list_images,
    load_model,
    make_lr_hr,
    parse_csv_list,
    parse_scales,
    scale_label,
)


FREQ_BINS = ("all", "low", "mid", "high")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_tensor(x: torch.Tensor, prefix: str) -> Dict[str, float]:
    vals = x.detach().float().flatten()
    if vals.numel() == 0:
        return {
            f"{prefix}_{k}": float("nan")
            for k in ("mean", "std", "min", "q05", "q50", "q95", "max")
        }
    vals_cpu = vals.cpu()
    if vals_cpu.numel() == 1:
        qs = vals_cpu.repeat(3)
    else:
        qs = torch.quantile(vals_cpu, torch.tensor([0.05, 0.50, 0.95]))
    return {
        f"{prefix}_mean": float(vals_cpu.mean()),
        f"{prefix}_std": float(vals_cpu.std(unbiased=False)),
        f"{prefix}_min": float(vals_cpu.min()),
        f"{prefix}_q05": float(qs[0]),
        f"{prefix}_q50": float(qs[1]),
        f"{prefix}_q95": float(qs[2]),
        f"{prefix}_max": float(vals_cpu.max()),
    }


def summarize_rows(
    rows: List[Dict[str, object]],
    keys: Sequence[str],
    metrics: Sequence[str],
) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, object]] = []
    for group_key, items in sorted(groups.items()):
        rec: Dict[str, object] = {k: v for k, v in zip(keys, group_key)}
        rec["n_rows"] = len(items)
        for metric in metrics:
            vals = [float(r[metric]) for r in items if r.get(metric) not in ("", None)]
            vals = [v for v in vals if math.isfinite(v)]
            rec[metric] = float(np.mean(vals)) if vals else float("nan")
        out.append(rec)
    return out


def make_query_coords(size: Sequence[int], max_queries: int, device: torch.device) -> torch.Tensor:
    coord = utils.make_coord(size).unsqueeze(0).to(device)
    if coord.shape[1] > max_queries:
        idx = torch.linspace(0, coord.shape[1] - 1, steps=max_queries, device=device).long()
        coord = coord[:, idx, :]
    return coord.contiguous()


def grid_fetch(feature_map: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    return (
        F.grid_sample(
            feature_map,
            coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :]
        .permute(0, 2, 1)
    )


def effective_omega_from_query(model, q_omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bs, q = q_omega.shape[:2]
    if hasattr(model, "tranNum") and hasattr(model, "num_freqs_per_tran"):
        t = int(model.tranNum)
        k = int(model.num_freqs_per_tran)
        omega = q_omega.view(bs, q, 2, k, t)
        omega_x = omega[:, :, 0]
        omega_y = omega[:, :, 1]
        cos_theta = model.cosTheta.detach().view(1, 1, 1, t)
        sin_theta = model.sinTheta.detach().view(1, 1, 1, t)
        eff_x = cos_theta * omega_x + sin_theta * omega_y
        eff_y = -sin_theta * omega_x + cos_theta * omega_y
        if getattr(model, "corrd_scale", 1.0) != 1.0:
            eff_x = eff_x * float(model.corrd_scale)
            eff_y = eff_y * float(model.corrd_scale)
        return eff_x.reshape(bs, q, k * t), eff_y.reshape(bs, q, k * t)

    k = int(model.num_freqs)
    omega = q_omega.view(bs, q, k, 2)
    return omega[..., 0], omega[..., 1]


def extract_effective_amplitude(
    model,
    coord: torch.Tensor,
    scale: float,
    lr_shape_hw: Sequence[int],
) -> Dict[str, torch.Tensor]:
    if not (hasattr(model, "coeff") and hasattr(model, "omega_map") and hasattr(model, "num_freqs")):
        raise ValueError("model does not expose SC-INR coefficient/omega maps")

    coord_ = coord.clamp(-1 + 1e-6, 1 - 1e-6)
    q_coef = grid_fetch(model.coeff, coord_)
    q_omega = grid_fetch(model.omega_map, coord_)
    q_coord = grid_fetch(model.feat_coord, coord_)

    rel_coord = coord - q_coord
    rel_coord[:, :, 0] *= model.feat.shape[-2]
    rel_coord[:, :, 1] *= model.feat.shape[-1]

    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / (int(lr_shape_hw[0]) * float(scale))
    cell[:, :, 1] *= 2 / (int(lr_shape_hw[1]) * float(scale))
    rel_cell = cell.clone()
    rel_cell[:, :, 0] *= model.feat.shape[-2]
    rel_cell[:, :, 1] *= model.feat.shape[-1]

    omega_x, omega_y = effective_omega_from_query(model, q_omega)
    k = int(model.num_freqs)
    coef_cos = q_coef[:, :, :k]
    coef_sin = q_coef[:, :, k : 2 * k]
    coef_pair_mag = torch.sqrt(coef_cos.square() + coef_sin.square() + 1e-12)

    c_x = rel_cell[:, :, 0:1]
    c_y = rel_cell[:, :, 1:2]
    response = torch.sinc(omega_x * c_x / 2) * torch.sinc(omega_y * c_y / 2)
    if bool(getattr(model, "use_sinc_response", True)):
        active_response = response
    else:
        active_response = torch.ones_like(response)

    eff_cos = coef_cos * active_response
    eff_sin = coef_sin * active_response
    effective_pair_mag = torch.sqrt(eff_cos.square() + eff_sin.square() + 1e-12)
    omega_mag = torch.sqrt(omega_x.square() + omega_y.square() + 1e-12)

    return {
        "coef_pair_mag": coef_pair_mag,
        "omega_mag": omega_mag,
        "response": response,
        "active_response": active_response,
        "effective_pair_mag": effective_pair_mag,
        "effective_input_energy": eff_cos.square() + eff_sin.square(),
        "coef_input_energy": coef_cos.square() + coef_sin.square(),
    }


def frequency_masks(omega_mag: torch.Tensor) -> Dict[str, torch.Tensor]:
    vals = omega_mag.detach().flatten()
    masks: Dict[str, torch.Tensor] = {"all": torch.ones_like(vals, dtype=torch.bool)}
    if vals.numel() < 3 or float(vals.min()) == float(vals.max()):
        masks["low"] = masks["all"]
        masks["mid"] = torch.zeros_like(vals, dtype=torch.bool)
        masks["high"] = torch.zeros_like(vals, dtype=torch.bool)
        return masks
    q1, q2 = torch.quantile(vals.float().cpu(), torch.tensor([1 / 3, 2 / 3])).to(vals.device)
    masks["low"] = vals <= q1
    masks["mid"] = (vals > q1) & (vals <= q2)
    masks["high"] = vals > q2
    return masks


def summarize_components(
    model_name: str,
    dataset: str,
    image: str,
    lr_scale: float,
    observation_scale: float,
    use_sinc_response: bool,
    components: Dict[str, torch.Tensor],
) -> List[Dict[str, object]]:
    flat = {key: value.detach().float().flatten() for key, value in components.items()}
    masks = frequency_masks(components["omega_mag"])
    rows: List[Dict[str, object]] = []
    for bin_name in FREQ_BINS:
        mask = masks[bin_name]
        n = int(mask.sum().item())
        rec: Dict[str, object] = {
            "model": model_name,
            "dataset": dataset,
            "image": image,
            "lr_scale": scale_label(lr_scale),
            "observation_scale": scale_label(observation_scale),
            "frequency_bin": bin_name,
            "n_values": n,
            "use_sinc_response": use_sinc_response,
        }
        if n == 0:
            for metric in (
                "omega_mag",
                "coef_pair_mag",
                "response",
                "response_abs",
                "active_response",
                "active_response_abs",
                "effective_pair_mag",
            ):
                rec.update({f"{metric}_{k}": float("nan") for k in ("mean", "std", "min", "q05", "q50", "q95", "max")})
            rec["response_neg_frac"] = float("nan")
            rec["active_response_neg_frac"] = float("nan")
            rec["effective_over_coef_mag_mean"] = float("nan")
            rec["effective_energy_ratio"] = float("nan")
            rows.append(rec)
            continue

        coef = flat["coef_pair_mag"][mask]
        eff = flat["effective_pair_mag"][mask]
        coef_energy = flat["coef_input_energy"][mask]
        eff_energy = flat["effective_input_energy"][mask]
        response = flat["response"][mask]
        active = flat["active_response"][mask]
        rec.update(summarize_tensor(flat["omega_mag"][mask], "omega_mag"))
        rec.update(summarize_tensor(coef, "coef_pair_mag"))
        rec.update(summarize_tensor(response, "response"))
        rec.update(summarize_tensor(response.abs(), "response_abs"))
        rec.update(summarize_tensor(active, "active_response"))
        rec.update(summarize_tensor(active.abs(), "active_response_abs"))
        rec.update(summarize_tensor(eff, "effective_pair_mag"))
        rec["response_neg_frac"] = float((response < 0).float().mean())
        rec["active_response_neg_frac"] = float((active < 0).float().mean())
        rec["effective_over_coef_mag_mean"] = float((eff / (coef + 1e-12)).mean())
        rec["effective_energy_ratio"] = float(eff_energy.mean() / (coef_energy.mean() + 1e-12))
        rows.append(rec)
    return rows


def plot_summary(summary_csv: Path, fig_dir: Path) -> None:
    if not summary_csv.exists():
        return
    import pandas as pd

    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    ensure_dir(fig_dir)
    plot_df = df[df["frequency_bin"].isin(["low", "mid", "high"])].copy()
    plot_df["scale_num"] = plot_df["observation_scale"].str.replace("x", "", regex=False).astype(float)
    for metric, ylabel, filename in [
        ("active_response_abs_mean", "Mean |active W|", "active_response_abs_by_freq.png"),
        ("effective_pair_mag_mean", "Mean effective coefficient-pair magnitude", "effective_pair_mag_by_freq.png"),
        ("effective_energy_ratio", "Effective/input coefficient energy ratio", "effective_energy_ratio_by_freq.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)
        for ax, bin_name in zip(axes, ["low", "mid", "high"]):
            sub_bin = plot_df[plot_df["frequency_bin"] == bin_name]
            for model_name, sub in sub_bin.groupby("model"):
                sub = sub.sort_values("scale_num")
                ax.plot(
                    sub["scale_num"],
                    sub[metric],
                    marker="o",
                    label=model_name,
                    color=STYLE.get(model_name),
                )
            ax.set_title(bin_name)
            ax.set_xlabel("Scale")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=220)
        plt.close(fig)


def run_analysis(args: argparse.Namespace) -> None:
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    rows: List[Dict[str, object]] = []
    for model_name in parse_csv_list(args.models):
        model = load_model(model_name, device)
        if not (hasattr(model, "coef") and hasattr(model, "omega_conv") and hasattr(model, "num_freqs")):
            print(f"[SKIP] {model_name}: not an SC-INR-style coefficient/omega decoder")
            continue
        for dataset in parse_csv_list(args.datasets):
            for img_path in list_images(dataset, args.max_images):
                lr, hr = make_lr_hr(img_path, args.lr_scale, device)
                coord = make_query_coords(hr.shape[-2:], args.max_queries, device)
                with torch.no_grad():
                    model.gen_feat((lr - 0.5) / 0.5)
                    for scale in parse_scales(args.scales):
                        comp = extract_effective_amplitude(model, coord, scale, lr.shape[-2:])
                        rows.extend(
                            summarize_components(
                                model_name,
                                dataset,
                                img_path.name,
                                args.lr_scale,
                                scale,
                                bool(getattr(model, "use_sinc_response", True)),
                                comp,
                            )
                        )
                del lr, hr, coord
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    write_csv(out_dir / "effective_amplitude_stats.csv", rows)
    metrics = [
        "omega_mag_mean",
        "coef_pair_mag_mean",
        "response_mean",
        "response_abs_mean",
        "active_response_mean",
        "active_response_abs_mean",
        "effective_pair_mag_mean",
        "response_neg_frac",
        "active_response_neg_frac",
        "effective_over_coef_mag_mean",
        "effective_energy_ratio",
    ]
    summary = summarize_rows(rows, ["model", "observation_scale", "frequency_bin"], metrics)
    write_csv(out_dir / "effective_amplitude_summary.csv", summary)
    plot_summary(out_dir / "effective_amplitude_summary.csv", out_dir / "figures")
    write_readme(out_dir, args)
    print(f"Effective-amplitude diagnostics written to {out_dir}")


def write_readme(out_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# Effective Amplitude Diagnostics

本目录诊断 SC-INR decoder 输入层面的 effective amplitude：

`q_coef(z) * W(omega(z), c)`。

这里的 `q_coef` 是送入 MLP 前、分别乘到 cos/sin 通道的 feature-level coefficient；
脚本用每个频率的 cos/sin coefficient pair magnitude 作为 `A(z)` 的可解释 proxy。
因此本结果不能解释为最终 RGB 图像频谱的严格振幅，原因是后续还有 MLP、local ensemble
和 residual upinput。

## 文件

- `effective_amplitude_stats.csv`：逐模型、数据集、图像、scale、频率分组的统计。
- `effective_amplitude_summary.csv`：按模型、scale、频率分组聚合。
- `figures/active_response_abs_by_freq.png`：不同频率组的 `|W|` 曲线。
- `figures/effective_pair_mag_by_freq.png`：不同频率组的 effective coefficient magnitude。
- `figures/effective_energy_ratio_by_freq.png`：`A_eff` 相对 `A` 的能量比例。
- `run_config.json`：运行参数。

## 覆盖范围

- models: `{args.models}`
- datasets: `{args.datasets}`
- max_images: `{args.max_images}`
- max_queries: `{args.max_queries}`
- scales: `{args.scales}`

如果这里是小样本配置或单图配置，本结果只能作为 mechanism sanity check，不能作为
数据集级趋势、模型排序或多 seed 证据。

## 解释边界

- `response` 是 signed sinc response；它可能为负，负号可视作 Fourier 分量的相位翻转。
- `active_response` 是模型实际使用的 response；对 `SC-INR-NoSinc` 它恒为 1。
- `effective_pair_mag` 使用 magnitude，因此反映的是输入强度变化，不保留 signed response 的符号。
- 本诊断是 `diagnostic-only`，需要与 benchmark、NoSinc 负控和 cell-intervention 一起解释。

## 参数

```json
{json.dumps(vars(args), indent=2, default=str)}
```
"""
    (out_dir / "README_zh.md").write_text(text)


def run_self_test() -> None:
    coef_cos = torch.tensor([1.0, 3.0, 0.0])
    coef_sin = torch.tensor([2.0, 4.0, 5.0])
    response = torch.tensor([1.0, -0.5, 0.0])
    coef_mag = torch.sqrt(coef_cos.square() + coef_sin.square() + 1e-12)
    eff_mag = torch.sqrt((coef_cos * response).square() + (coef_sin * response).square() + 1e-12)
    expected = coef_mag * response.abs()
    if not torch.allclose(eff_mag, expected, atol=1e-5):
        raise AssertionError("effective magnitude must equal coefficient magnitude times |W|")

    omega = torch.tensor([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
    masks = frequency_masks(omega)
    if int(masks["all"].sum()) != omega.numel():
        raise AssertionError("all frequency bin must include every value")
    if int(masks["low"].sum() + masks["mid"].sum() + masks["high"].sum()) != omega.numel():
        raise AssertionError("low/mid/high bins must partition values")
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SC-INR feature-level effective amplitude.")
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "derived" / "diagnostics" / "effective_amplitude_2026-05-12",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="SC-INR-NoPhi,SC-INR,SC-INR-NoSinc,SC-INR-EQ")
    parser.add_argument("--datasets", default="bsd100,urban100")
    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument("--lr_scale", type=int, default=4)
    parser.add_argument("--scales", default="2,3,4,6,8,12,16,24,30")
    parser.add_argument("--max_queries", type=int, default=2048)
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    run_analysis(args)


if __name__ == "__main__":
    main()
