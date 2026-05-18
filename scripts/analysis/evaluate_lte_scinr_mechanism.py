#!/usr/bin/env python3
"""统一诊断 LTE cell-phase 与 SC-INR footprint-response 机制。

本脚本服务论文机制链路，而不是替代 benchmark。它在同一批 LR、query 和
cell multiplier 下同时记录：

1. footprint oracle 指标：模型输出是否跟随 HR box-average proxy；
2. LTE 路径：单独隔离 learned cell-conditioned phase h_p(c)；
3. SC-INR 路径：记录 analytic W(omega,c) 以及 MLP 前 effective amplitude proxy；
4. 负控：cell-insensitive 模型是否因为不响应 cell 而刷高自一致性。

解释边界：
- SC-INR 的 effective amplitude 只指 MLP 前 Fourier feature input proxy；
- footprint oracle 是有限 HR 图像上的 piecewise-constant proxy；
- 本诊断不能证明 sinc 唯一因果、RGB exact box integral 或 strict scale equivariance。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import utils
from scripts.analysis.evaluate_footprint_oracle import (
    box_oracle,
)
from scripts.analysis.evaluate_seed1_aux_metrics import (
    DATASETS,
    ensure_dir,
    list_images,
    load_model,
    rgb_to_y,
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


def pil_to_tensor(img: Image.Image, device: torch.device | str) -> torch.Tensor:
    return transforms.ToTensor()(img.convert("RGB")).unsqueeze(0).to(device)


def make_lr_hr_for_diag(img_path: Path, scale: float, crop_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """生成诊断用 LR/HR；可选中心 crop 以控制机制实验成本。"""

    img_hr_pil = Image.open(img_path).convert("RGB")
    w_hr, h_hr = img_hr_pil.size
    if crop_size and crop_size > 0:
        crop = int(crop_size)
        crop = min(crop, w_hr, h_hr)
        crop = max(int(math.floor(crop / scale)) * int(scale), int(scale))
        left = max(0, (w_hr - crop) // 2)
        top = max(0, (h_hr - crop) // 2)
        img_hr_pil = img_hr_pil.crop((left, top, left + crop, top + crop))
        w_hr, h_hr = img_hr_pil.size
    h_lr = max(1, int(math.floor(h_hr / scale + 1e-9)))
    w_lr = max(1, int(math.floor(w_hr / scale + 1e-9)))
    target_h = int(round(h_lr * scale))
    target_w = int(round(w_lr * scale))
    hr_crop = img_hr_pil.crop((0, 0, target_w, target_h))
    lr = hr_crop.resize((w_lr, h_lr), Image.BICUBIC)
    return pil_to_tensor(lr, device), pil_to_tensor(hr_crop, device)


def finite_mean(vals: Sequence[float]) -> float:
    finite = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


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
            rec[metric] = finite_mean([float(r.get(metric, float("nan"))) for r in items])
        out.append(rec)
    return out


def sample_coord(h: int, w: int, max_queries: int, device: torch.device | str, shave: int = 0) -> torch.Tensor:
    full = utils.make_coord([h, w]).view(h, w, 2)
    if shave > 0 and h > 2 * shave and w > 2 * shave:
        full = full[shave : h - shave, shave : w - shave, :]
    coord = full.reshape(1, -1, 2).to(device)
    if max_queries > 0 and coord.shape[1] > max_queries:
        idx = torch.linspace(0, coord.shape[1] - 1, steps=max_queries, device=device).long()
        coord = coord[:, idx, :]
    return coord.contiguous()


def make_cell_like(coord: torch.Tensor, h: int, w: int, multiplier: float) -> torch.Tensor:
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2.0 * float(multiplier) / h
    cell[:, :, 1] *= 2.0 * float(multiplier) / w
    return cell.contiguous()


def query_oracle(oracle: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    return (
        F.grid_sample(oracle, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False)[:, :, 0, :]
        .permute(0, 2, 1)
        .contiguous()
    )


def query_model_with_cell(model, coord: torch.Tensor, cell: torch.Tensor, bsize: int) -> torch.Tensor:
    preds = []
    with torch.no_grad():
        for ql in range(0, coord.shape[1], bsize):
            qr = min(ql + bsize, coord.shape[1])
            preds.append(model.query_rgb(coord[:, ql:qr], cell[:, ql:qr]))
    pred = torch.cat(preds, dim=1)
    return (pred * 0.5 + 0.5).clamp(0, 1)


def query_metric_dict(
    pred: torch.Tensor,
    pred_base: torch.Tensor,
    oracle: torch.Tensor,
    oracle_base: torch.Tensor,
    multiplier: float,
    weak_threshold: float,
) -> Dict[str, object]:
    pred_y = rgb_to_y(pred.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
    base_y = rgb_to_y(pred_base.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
    oracle_y = rgb_to_y(oracle.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
    oracle_base_y = rgb_to_y(oracle_base.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

    oracle_diff = pred_y - oracle_y
    oracle_mse = float(oracle_diff.square().mean().item())
    pred_delta = pred_y - base_y
    oracle_delta = oracle_y - oracle_base_y
    tracking = pred_delta - oracle_delta
    delta_tracking_rmse = float(torch.sqrt(tracking.square().mean()).item())
    cell_sensitivity_rmse = float(torch.sqrt(pred_delta.square().mean()).item())
    oracle_change_rmse = float(torch.sqrt(oracle_delta.square().mean()).item())
    oracle_psnr = -10.0 * math.log10(max(oracle_mse, 1e-12))
    return {
        "oracle_psnr_y": oracle_psnr,
        "oracle_rmse_y": math.sqrt(oracle_mse),
        "delta_tracking_rmse_y": delta_tracking_rmse,
        "cell_sensitivity_rmse_y": cell_sensitivity_rmse,
        "oracle_change_rmse_y": oracle_change_rmse,
        "weak_oracle": oracle_change_rmse < weak_threshold and abs(float(multiplier) - 1.0) > 1e-12,
        "valid_shave": 0,
    }


def grid_fetch(feature_map: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    return (
        F.grid_sample(
            feature_map,
            coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :]
        .permute(0, 2, 1)
        .contiguous()
    )


def rmse(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return float("nan")
    n = min(a.numel(), b.numel())
    if n == 0:
        return float("nan")
    return float(torch.sqrt((a.flatten()[:n] - b.flatten()[:n]).float().square().mean()).item())


def tensor_mean(x: torch.Tensor | None) -> float:
    return float(x.detach().float().mean().item()) if x is not None and x.numel() else float("nan")


def tensor_rms(x: torch.Tensor | None) -> float:
    return float(torch.sqrt(x.detach().float().square().mean()).item()) if x is not None and x.numel() else float("nan")


def tensor_neg_frac(x: torch.Tensor | None) -> float:
    return float((x.detach().float() < 0).float().mean().item()) if x is not None and x.numel() else float("nan")


def rel_cell_for_model(model, cell: torch.Tensor) -> torch.Tensor:
    feat = getattr(model, "feat", None)
    if feat is None:
        feat = getattr(model, "coeff", None)
    if feat is None:
        raise RuntimeError("model must call gen_feat before signal extraction")
    rel_cell = cell.clone()
    rel_cell[:, :, 0] *= feat.shape[-2]
    rel_cell[:, :, 1] *= feat.shape[-1]
    return rel_cell


def extract_lte_signals(model, coord: torch.Tensor, cell: torch.Tensor) -> Dict[str, torch.Tensor | None]:
    """提取 LTE-family 中与 phase/cell 相关的 MLP 前信号。

    这里故意只使用中心 nearest feature，不展开 local ensemble。目的是隔离机制信号，
    而不是复刻完整 query_rgb 的加权输出。
    """

    if not (hasattr(model, "coeff") and hasattr(model, "freqq")):
        return {}

    feat = model.feat
    rel_cell = rel_cell_for_model(model, cell)
    q_coef = grid_fetch(model.coeff, coord)
    q_freq = grid_fetch(model.freqq, coord)
    q_coord = grid_fetch(model.feat_coord, coord)
    rel_coord = coord - q_coord
    rel_coord[:, :, 0] *= feat.shape[-2]
    rel_coord[:, :, 1] *= feat.shape[-1]

    bs, q = coord.shape[:2]
    q_freq_pair = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
    base_phase = torch.sum(q_freq_pair * rel_coord.unsqueeze(-1), dim=-2)

    cell_phase = None
    feature_phase = None
    phase_type = "no_phase"
    if hasattr(model, "phase"):
        cell_phase = model.phase(rel_cell.reshape(bs * q, -1)).reshape(bs, q, -1)
        phase = base_phase + cell_phase
        phase_type = "cell_conditioned_phase"
    elif hasattr(model, "phase_map"):
        feature_phase = grid_fetch(model.phase_map, coord)
        phase = base_phase + feature_phase
        phase_type = "feature_conditioned_phase"
    else:
        phase = base_phase

    basis = torch.cat([torch.cos(math.pi * phase), torch.sin(math.pi * phase)], dim=-1)
    imnet_input = q_coef * basis
    k = imnet_input.shape[-1] // 2
    imnet_pair_mag = torch.sqrt(imnet_input[..., :k].square() + imnet_input[..., k:].square())

    return {
        "path_type": phase_type,  # type: ignore[dict-item]
        "cell_phase": cell_phase,
        "feature_phase": feature_phase,
        "imnet_pair_mag": imnet_pair_mag,
    }


def extract_sc_signals(model, coord: torch.Tensor, cell: torch.Tensor) -> Dict[str, torch.Tensor | None]:
    """提取 SC-INR-family 的 response 与 MLP 前 amplitude proxy。"""

    if not (hasattr(model, "omega_map") and hasattr(model, "coeff") and hasattr(model, "num_freqs")):
        return {}

    feat = model.feat
    rel_cell = rel_cell_for_model(model, cell)
    q_coef = grid_fetch(model.coeff, coord)
    q_omega = grid_fetch(model.omega_map, coord)
    q_coord = grid_fetch(model.feat_coord, coord)
    rel_coord = coord - q_coord
    rel_coord[:, :, 0] *= feat.shape[-2]
    rel_coord[:, :, 1] *= feat.shape[-1]

    bs, q = coord.shape[:2]
    k = int(model.num_freqs)
    q_omega = q_omega.view(bs, q, k, 2)
    phase = torch.sum(q_omega * rel_coord.unsqueeze(-2), dim=-1)
    feature_phase = None
    if getattr(model, "phase_map", None) is not None:
        feature_phase = grid_fetch(model.phase_map, coord)
        phase = phase + feature_phase

    omega_x = q_omega[:, :, :, 0]
    omega_y = q_omega[:, :, :, 1]
    c_x = rel_cell[:, :, 0:1]
    c_y = rel_cell[:, :, 1:2]
    analytic_response = torch.sinc(omega_x * c_x / 2) * torch.sinc(omega_y * c_y / 2)
    if bool(getattr(model, "use_sinc_response", True)):
        active_response = analytic_response
    else:
        active_response = torch.ones_like(analytic_response)

    coef_cos = q_coef[..., :k]
    coef_sin = q_coef[..., k:]
    coef_pair_mag = torch.sqrt(coef_cos.square() + coef_sin.square())
    effective_pair_mag = coef_pair_mag * active_response.abs()
    effective_energy_ratio = effective_pair_mag.square() / (coef_pair_mag.square() + 1e-12)

    basis_cos = torch.cos(math.pi * phase) * active_response
    basis_sin = torch.sin(math.pi * phase) * active_response
    imnet_input_cos = coef_cos * basis_cos
    imnet_input_sin = coef_sin * basis_sin
    imnet_pair_mag = torch.sqrt(imnet_input_cos.square() + imnet_input_sin.square())

    return {
        "path_type": "analytic_sinc_response" if bool(getattr(model, "use_sinc_response", True)) else "no_sinc_response",  # type: ignore[dict-item]
        "feature_phase": feature_phase,
        "analytic_response": analytic_response,
        "active_response": active_response,
        "coef_pair_mag": coef_pair_mag,
        "effective_pair_mag": effective_pair_mag,
        "effective_energy_ratio": effective_energy_ratio,
        "imnet_pair_mag": imnet_pair_mag,
    }


def extract_signals(model_name: str, model, coord: torch.Tensor, cell: torch.Tensor) -> Dict[str, object]:
    signals: Dict[str, object] = {"model": model_name, "path_type": "unsupported"}
    sc = extract_sc_signals(model, coord, cell)
    if sc:
        signals.update(sc)
        return signals
    lte = extract_lte_signals(model, coord, cell)
    if lte:
        signals.update(lte)
        return signals
    return signals


def signal_metrics(current: Dict[str, object], base: Dict[str, object]) -> Dict[str, float]:
    cell_phase = current.get("cell_phase")
    feature_phase = current.get("feature_phase")
    analytic_response = current.get("analytic_response")
    active_response = current.get("active_response")
    effective_pair_mag = current.get("effective_pair_mag")
    effective_energy_ratio = current.get("effective_energy_ratio")
    imnet_pair_mag = current.get("imnet_pair_mag")

    return {
        "cell_phase_rms": tensor_rms(cell_phase if isinstance(cell_phase, torch.Tensor) else None),
        "cell_phase_delta_rms_vs_native": rmse(
            cell_phase if isinstance(cell_phase, torch.Tensor) else None,
            base.get("cell_phase") if isinstance(base.get("cell_phase"), torch.Tensor) else None,
        ),
        "feature_phase_rms": tensor_rms(feature_phase if isinstance(feature_phase, torch.Tensor) else None),
        "feature_phase_delta_rms_vs_native": rmse(
            feature_phase if isinstance(feature_phase, torch.Tensor) else None,
            base.get("feature_phase") if isinstance(base.get("feature_phase"), torch.Tensor) else None,
        ),
        "analytic_response_mean": tensor_mean(analytic_response if isinstance(analytic_response, torch.Tensor) else None),
        "analytic_response_abs_mean": tensor_mean(
            analytic_response.abs() if isinstance(analytic_response, torch.Tensor) else None
        ),
        "analytic_response_neg_frac": tensor_neg_frac(analytic_response if isinstance(analytic_response, torch.Tensor) else None),
        "active_response_mean": tensor_mean(active_response if isinstance(active_response, torch.Tensor) else None),
        "active_response_abs_mean": tensor_mean(active_response.abs() if isinstance(active_response, torch.Tensor) else None),
        "active_response_neg_frac": tensor_neg_frac(active_response if isinstance(active_response, torch.Tensor) else None),
        "active_response_delta_rms_vs_native": rmse(
            active_response if isinstance(active_response, torch.Tensor) else None,
            base.get("active_response") if isinstance(base.get("active_response"), torch.Tensor) else None,
        ),
        "effective_pair_mag_mean": tensor_mean(
            effective_pair_mag if isinstance(effective_pair_mag, torch.Tensor) else None
        ),
        "effective_pair_delta_rms_vs_native": rmse(
            effective_pair_mag if isinstance(effective_pair_mag, torch.Tensor) else None,
            base.get("effective_pair_mag") if isinstance(base.get("effective_pair_mag"), torch.Tensor) else None,
        ),
        "effective_energy_ratio_mean": tensor_mean(
            effective_energy_ratio if isinstance(effective_energy_ratio, torch.Tensor) else None
        ),
        "imnet_input_pair_mag_mean": tensor_mean(imnet_pair_mag if isinstance(imnet_pair_mag, torch.Tensor) else None),
        "imnet_input_pair_delta_rms_vs_native": rmse(
            imnet_pair_mag if isinstance(imnet_pair_mag, torch.Tensor) else None,
            base.get("imnet_pair_mag") if isinstance(base.get("imnet_pair_mag"), torch.Tensor) else None,
        ),
    }


def plot_overall(overall_path: Path, fig_dir: Path) -> None:
    if not overall_path.exists():
        return
    import pandas as pd

    df = pd.read_csv(overall_path)
    if df.empty:
        return
    ensure_dir(fig_dir)
    df["cell_multiplier_num"] = df["cell_multiplier"].astype(float)
    df = df[df["cell_multiplier_num"] > 1.0].copy()
    if df.empty:
        return

    plots = [
        ("delta_tracking_rmse_y", "Footprint delta-tracking RMSE-Y", "footprint_tracking.png"),
        ("cell_sensitivity_rmse_y", "Output cell sensitivity RMSE-Y", "output_cell_sensitivity.png"),
        ("cell_phase_delta_rms_vs_native", "LTE h_p(c) delta RMS", "cell_phase_delta.png"),
        ("active_response_delta_rms_vs_native", "SC-INR active response delta RMS", "active_response_delta.png"),
        ("effective_pair_delta_rms_vs_native", "SC-INR effective amplitude proxy delta RMS", "effective_pair_delta.png"),
    ]
    for metric, ylabel, filename in plots:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for model_name, sub in df.groupby("model"):
            sub = sub.sort_values("cell_multiplier_num")
            vals = sub[metric].astype(float)
            if vals.notna().sum() == 0:
                continue
            ax.plot(
                sub["cell_multiplier_num"],
                vals,
                marker="o",
                color=STYLE.get(model_name, "#888888"),
                label=model_name,
            )
        ax.set_xlabel("Cell multiplier")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=220)
        plt.close(fig)


def gate_conclusion(overall_rows: List[Dict[str, object]]) -> Dict[str, object]:
    rows = [r for r in overall_rows if float(r["cell_multiplier"]) > 1.0]

    def mean_metric(model: str, metric: str) -> float:
        vals = [float(r[metric]) for r in rows if r["model"] == model and math.isfinite(float(r.get(metric, float("nan"))))]
        return finite_mean(vals)

    sc_track = mean_metric("SC-INR", "delta_tracking_rmse_y")
    lte_track = mean_metric("LTE", "delta_tracking_rmse_y")
    nosinc_track = mean_metric("SC-INR-NoSinc", "delta_tracking_rmse_y")
    sc_oracle = mean_metric("SC-INR", "oracle_rmse_y")
    lte_oracle = mean_metric("LTE", "oracle_rmse_y")
    nosinc_oracle = mean_metric("SC-INR-NoSinc", "oracle_rmse_y")
    lte_phase_delta = mean_metric("LTE", "cell_phase_delta_rms_vs_native")
    sc_response_delta = mean_metric("SC-INR", "active_response_delta_rms_vs_native")
    nosinc_response_delta = mean_metric("SC-INR-NoSinc", "active_response_delta_rms_vs_native")

    supports_footprint_vs_nosinc = (
        math.isfinite(sc_track)
        and math.isfinite(nosinc_track)
        and sc_track < nosinc_track
        and (not math.isfinite(nosinc_oracle) or sc_oracle <= nosinc_oracle * 1.02)
    )
    supports_footprint_vs_lte = (
        math.isfinite(sc_track)
        and math.isfinite(lte_track)
        and sc_track < lte_track
        and math.isfinite(sc_oracle)
        and math.isfinite(lte_oracle)
        and sc_oracle <= lte_oracle
    )
    has_lte_phase_signal = math.isfinite(lte_phase_delta) and lte_phase_delta > 0
    has_sc_response_signal = math.isfinite(sc_response_delta) and sc_response_delta > 0
    nosinc_is_cell_insensitive = math.isfinite(nosinc_response_delta) and nosinc_response_delta < 1e-12

    return {
        "supports_footprint_vs_nosinc": supports_footprint_vs_nosinc,
        "supports_footprint_vs_lte": supports_footprint_vs_lte,
        "has_lte_cell_phase_signal": has_lte_phase_signal,
        "has_sc_inr_active_response_signal": has_sc_response_signal,
        "nosinc_active_response_cell_insensitive": nosinc_is_cell_insensitive,
        "mean_metrics_m_gt_1": {
            "SC-INR_delta_tracking_rmse_y": sc_track,
            "LTE_delta_tracking_rmse_y": lte_track,
            "SC-INR-NoSinc_delta_tracking_rmse_y": nosinc_track,
            "SC-INR_oracle_rmse_y": sc_oracle,
            "LTE_oracle_rmse_y": lte_oracle,
            "SC-INR-NoSinc_oracle_rmse_y": nosinc_oracle,
            "LTE_cell_phase_delta_rms_vs_native": lte_phase_delta,
            "SC-INR_active_response_delta_rms_vs_native": sc_response_delta,
            "SC-INR-NoSinc_active_response_delta_rms_vs_native": nosinc_response_delta,
        },
        "claim_boundary": (
            "该 gate 只支持同一 LR/query 下的机制差异：LTE 有 learned h_p(c) cell-phase 信号，"
            "SC-INR 有非零 analytic response/effective-amplitude proxy，且 footprint oracle 上可与 "
            "LTE/NoSinc 比较。它不能证明 sinc 唯一因果或最终 RGB exact box integral。"
        ),
    }


def write_readme(out_dir: Path, args: argparse.Namespace, conclusion: Dict[str, object]) -> None:
    text = f"""# LTE vs SC-INR 机制诊断 2026-05-16

本目录把 `LTE -> SC-INR` 主脉络中的机制证据放到同一个协议下检查：

- 固定 x{args.base_scale} LR 和 query grid；
- 只改变 decoder 输入 cell multiplier：`{args.cell_multipliers}`；
- 同时记录 footprint oracle、输出 cell sensitivity、LTE `h_p(c)`、SC-INR `W(omega,c)`、
  以及 MLP 前 effective amplitude proxy。

## 覆盖

- models: `{args.models}`
- datasets: `{args.datasets}`
- max_images: `{args.max_images}`
- crop size: `{args.crop_size}`，`0` 表示不用 crop。
- signal/oracle max queries: `{args.max_signal_queries}`

## 主要输出

- `mechanism_metrics.csv`：逐图、逐模型、逐 cell multiplier 指标。
- `mechanism_summary.csv`：按 dataset/model/cell multiplier 汇总。
- `mechanism_overall.csv`：按 model/cell multiplier 汇总。
- `gate_conclusion.json`：自动 gate 摘要。
- `figures/`：小型机制曲线图。

## Gate 结论

```json
{json.dumps(conclusion, indent=2, ensure_ascii=False)}
```

## 解释边界

- `cell_phase_*` 只对 LTE 的 learned `h_p(c)` 有直接语义；`LTE-NoCellPhase`
  应为无 cell phase，`LTE-PhaseZ` 的 feature phase 不随 cell 变化。
- `effective_pair_*` 是 SC-INR MLP 前 Fourier input 的 coefficient-pair magnitude proxy，
  不能解释为最终 RGB 频谱振幅。
- footprint oracle 是有限 HR 上的 piecewise-constant box proxy，不是真实连续场景积分。
- 如果本目录结果与 benchmark 冲突，应优先收缩 claim，而不是把诊断指标升级为主证据。
"""
    (out_dir / "README_zh.md").write_text(text)


def run_analysis(args: argparse.Namespace) -> None:
    out_dir = args.out
    ensure_dir(out_dir / "figures")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model_names = parse_csv_list(args.models)
    missing = [m for m in model_names if m not in MODEL_PATHS or not MODEL_PATHS[m].exists()]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints: {missing}")

    multipliers = parse_float_list(args.cell_multipliers)
    if 1.0 not in multipliers:
        raise ValueError("--cell_multipliers must include 1 for native-cell reference")

    rows: List[Dict[str, object]] = []
    loaded = {name: load_model(name, device) for name in model_names}
    try:
        for dataset in parse_csv_list(args.datasets):
            for img_path in list_images(dataset, args.max_images):
                lr, hr = make_lr_hr_for_diag(img_path, args.base_scale, args.crop_size, device)
                h, w = hr.shape[-2:]
                max_shave = max(int(args.base_scale), int(math.ceil(max(multipliers) / 2.0)) + 1)
                signal_coord = sample_coord(h, w, args.max_signal_queries, device, shave=max_shave)
                signal_cells = {m: make_cell_like(signal_coord, h, w, m) for m in multipliers}
                oracles = {m: box_oracle(hr, m).clamp(0, 1) for m in multipliers}
                oracle_queries = {m: query_oracle(oracles[m], signal_coord) for m in multipliers}
                oracle_base = oracle_queries[1.0]

                for model_name, model in loaded.items():
                    inp = (lr - 0.5) / 0.5
                    with torch.no_grad():
                        model.gen_feat(inp)
                        preds = {m: query_model_with_cell(model, signal_coord, signal_cells[m], args.eval_bsize) for m in multipliers}
                        signals = {
                            m: extract_signals(model_name, model, signal_coord, signal_cells[m])
                            for m in multipliers
                        }
                    pred_base = preds[1.0]
                    signal_base = signals[1.0]

                    for multiplier in multipliers:
                        rec: Dict[str, object] = {
                            "model": model_name,
                            "path_type": str(signals[multiplier].get("path_type", "unsupported")),
                            "dataset": dataset,
                            "image": img_path.name,
                            "base_scale": f"x{args.base_scale}",
                            "cell_multiplier": f"{multiplier:g}",
                        }
                        rec.update(
                            query_metric_dict(
                                preds[multiplier],
                                pred_base,
                                oracle_queries[multiplier],
                                oracle_base,
                                multiplier,
                                args.weak_oracle_threshold,
                            )
                        )
                        rec.update(signal_metrics(signals[multiplier], signal_base))
                        rows.append(rec)

                    del preds, signals
                    torch.cuda.empty_cache()
                del lr, hr, oracles, oracle_queries, oracle_base, signal_coord, signal_cells
                torch.cuda.empty_cache()
    finally:
        del loaded
        torch.cuda.empty_cache()

    metrics = [
        "oracle_psnr_y",
        "oracle_rmse_y",
        "delta_tracking_rmse_y",
        "cell_sensitivity_rmse_y",
        "oracle_change_rmse_y",
        "cell_phase_rms",
        "cell_phase_delta_rms_vs_native",
        "feature_phase_rms",
        "feature_phase_delta_rms_vs_native",
        "analytic_response_mean",
        "analytic_response_abs_mean",
        "analytic_response_neg_frac",
        "active_response_mean",
        "active_response_abs_mean",
        "active_response_neg_frac",
        "active_response_delta_rms_vs_native",
        "effective_pair_mag_mean",
        "effective_pair_delta_rms_vs_native",
        "effective_energy_ratio_mean",
        "imnet_input_pair_mag_mean",
        "imnet_input_pair_delta_rms_vs_native",
    ]
    write_csv(out_dir / "mechanism_metrics.csv", rows)
    summary = summarize(rows, ["dataset", "model", "path_type", "cell_multiplier"], metrics)
    overall = summarize(rows, ["model", "path_type", "cell_multiplier"], metrics)
    write_csv(out_dir / "mechanism_summary.csv", summary)
    write_csv(out_dir / "mechanism_overall.csv", overall)
    plot_overall(out_dir / "mechanism_overall.csv", out_dir / "figures")
    conclusion = gate_conclusion(overall)
    (out_dir / "gate_conclusion.json").write_text(json.dumps(conclusion, indent=2, ensure_ascii=False, default=str))
    write_readme(out_dir, args, conclusion)
    print(f"LTE-vs-SC-INR mechanism diagnostics written to {out_dir}")
    print(json.dumps(conclusion, indent=2, ensure_ascii=False, default=str))


def run_self_test() -> None:
    base = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    cur = base + 0.5
    got = rmse(cur, base)
    if abs(got - 0.5) > 1e-7:
        raise AssertionError(f"rmse helper failed: {got}")

    omega = torch.tensor([0.25])
    w1 = torch.sinc(omega * 1.0 / 2)
    w4 = torch.sinc(omega * 4.0 / 2)
    if not float(w4) < float(w1):
        raise AssertionError("larger footprint should attenuate this low-frequency sinusoid more")

    const = torch.ones(1, 3, 12, 13)
    if not torch.allclose(box_oracle(const, 4.0), const, atol=1e-7):
        raise AssertionError("footprint oracle must preserve constants")
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LTE-vs-SC-INR phase/response mechanism diagnostics.")
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--models", default="LTE,LTE-NoCellPhase,LTE-PhaseZ,SC-INR-NoPhi,SC-INR,SC-INR-NoSinc")
    parser.add_argument("--datasets", default="bsd100,urban100")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--base_scale", type=int, default=4)
    parser.add_argument("--cell_multipliers", default="1,2,4")
    parser.add_argument("--weak_oracle_threshold", type=float, default=0.002)
    parser.add_argument("--max_signal_queries", type=int, default=4096)
    parser.add_argument("--crop_size", type=int, default=192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_bsize", type=int, default=50000)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "derived" / "diagnostics" / "lte_scinr_mechanism_2026-05-16",
    )
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    os.chdir(ROOT)
    run_analysis(args)


if __name__ == "__main__":
    main()
