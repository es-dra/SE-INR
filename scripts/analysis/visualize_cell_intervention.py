#!/usr/bin/env python3
"""Cell intervention visualization for SC-INR mechanism demonstration.

固定同一个 LR crop 和同一组 query 坐标，只改变 decoder 输入的 cell multiplier，
比较 LIIF/LTE/LTE-PhaseZ/SC-INR/SC-INR-NoSinc 的输出变化。

本脚本不是 benchmark，也不是新的训练实验。它用于周报/论文机制展示：

- LTE: cell 可能通过 learned phase 造成纹理相位式变化；
- SC-INR-NoSinc: 去掉 analytic response 后几乎不响应 cell；
- SC-INR: 保留非零 footprint response，更接近 HR box-average oracle 的变化。

候选 crop 只由 GT 的高频/边缘强度预筛，再按预先定义的 tracking 指标排序，
避免手工只挑最有利图。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parent / "Data"
sys.path.insert(0, str(ROOT))

import utils  # noqa: E402
from scripts.analysis.evaluate_footprint_oracle import box_oracle  # noqa: E402
from scripts.analysis.evaluate_seed1_aux_metrics import load_model, rgb_to_y  # noqa: E402
from scripts.analysis.model_registry import STYLE  # noqa: E402


DATASETS = {
    "bsd100": DATA_ROOT / "BSD100" / "HR",
    "urban100": DATA_ROOT / "Urban100" / "HR",
}
DEFAULT_MODELS = ["LIIF", "LTE", "LTE-PhaseZ", "SC-INR", "SC-INR-NoSinc"]
PLOT_MODELS = ["Oracle", "LIIF", "LTE", "LTE-PhaseZ", "SC-INR", "SC-INR-NoSinc"]
DISPLAY_NAMES = {
    "Oracle": "HR box oracle",
    "LIIF": "LIIF",
    "LTE": "LTE",
    "LTE-PhaseZ": "LTE-PhaseZ",
    "SC-INR": "SC-INR",
    "SC-INR-NoSinc": "SC-INR-NoSinc",
}


@dataclass(frozen=True)
class CropCandidate:
    dataset: str
    image: str
    path: Path
    x: int
    y: int
    size: int
    texture_score: float


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def tensor_to_img(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return arr


def list_images(dataset: str, max_images: int) -> List[Path]:
    root = DATASETS[dataset]
    paths = sorted([p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def crop_texture_score(img: Image.Image) -> float:
    y = transforms.ToTensor()(img.convert("RGB")).unsqueeze(0)
    coeff = y.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
    yy = (y * coeff).sum(dim=1, keepdim=True)
    lap = yy.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
    hp = F.conv2d(yy, lap, padding=1)
    var = F.avg_pool2d(yy.square(), 7, stride=1, padding=3) - F.avg_pool2d(yy, 7, stride=1, padding=3).square()
    return float(torch.sqrt(hp.square().mean()).item() + 0.5 * torch.sqrt(var.clamp_min(0).mean()).item())


def enumerate_candidates(
    datasets: Sequence[str],
    max_images: int,
    crop_size: int,
    stride: int,
    max_crops_per_image: int,
    base_scale: int,
) -> List[CropCandidate]:
    candidates: List[CropCandidate] = []
    crop_size = int(math.floor(crop_size / base_scale)) * base_scale
    stride = int(math.floor(stride / base_scale)) * base_scale
    for dataset in datasets:
        for path in list_images(dataset, max_images):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            local: List[CropCandidate] = []
            if w < crop_size or h < crop_size:
                continue
            xs = list(range(0, max(1, w - crop_size + 1), stride))
            ys = list(range(0, max(1, h - crop_size + 1), stride))
            if xs[-1] != w - crop_size:
                xs.append(w - crop_size)
            if ys[-1] != h - crop_size:
                ys.append(h - crop_size)
            for y in ys:
                for x in xs:
                    crop = img.crop((x, y, x + crop_size, y + crop_size))
                    score = crop_texture_score(crop)
                    local.append(CropCandidate(dataset, path.name, path, x, y, crop_size, score))
            local.sort(key=lambda c: c.texture_score, reverse=True)
            candidates.extend(local[:max_crops_per_image])
    candidates.sort(key=lambda c: c.texture_score, reverse=True)
    return candidates


def make_lr_hr_crop(candidate: CropCandidate, base_scale: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    img = Image.open(candidate.path).convert("RGB")
    hr_pil = img.crop((candidate.x, candidate.y, candidate.x + candidate.size, candidate.y + candidate.size))
    lr_size = candidate.size // base_scale
    lr_pil = hr_pil.resize((lr_size, lr_size), Image.BICUBIC)
    return pil_to_tensor(lr_pil, device), pil_to_tensor(hr_pil, device)


def make_coord_cell(h: int, w: int, multiplier: float, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    coord = utils.make_coord([h, w]).unsqueeze(0).to(device)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2.0 * float(multiplier) / h
    cell[:, :, 1] *= 2.0 * float(multiplier) / w
    return coord.contiguous(), cell.contiguous()


def predict_with_cell(model, lr: torch.Tensor, multiplier: float, bsize: int, base_scale: int) -> torch.Tensor:
    device = lr.device
    h = lr.shape[-2] * int(base_scale)
    w = lr.shape[-1] * int(base_scale)
    inp = (lr - 0.5) / 0.5
    coord, cell = make_coord_cell(h, w, multiplier, device)
    with torch.no_grad():
        model.gen_feat(inp)
        preds = []
        for ql in range(0, coord.shape[1], bsize):
            qr = min(ql + bsize, coord.shape[1])
            preds.append(model.query_rgb(coord[:, ql:qr], cell[:, ql:qr]))
        pred = torch.cat(preds, dim=1)
    pred = (pred * 0.5 + 0.5).clamp(0, 1)
    return pred.view(1, h, w, 3).permute(0, 3, 1, 2).contiguous()


def valid_y(x: torch.Tensor, shave: int) -> torch.Tensor:
    y = rgb_to_y(x)
    if shave > 0 and y.shape[-2] > 2 * shave and y.shape[-1] > 2 * shave:
        y = y[..., shave:-shave, shave:-shave]
    return y


def rmse(x: torch.Tensor) -> float:
    return float(torch.sqrt(x.float().square().mean()).item())


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.float().flatten()
    bv = b.float().flatten()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = torch.sqrt(av.square().mean() * bv.square().mean()).item()
    if denom < 1e-12:
        return float("nan")
    return float((av * bv).mean().item() / denom)


def highpass_rms(x: torch.Tensor) -> float:
    y = valid_y(x, 0)
    kernel = y.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
    hp = F.conv2d(y, kernel, padding=1)
    return rmse(hp)


def best_delta_shift(
    pred_delta: torch.Tensor,
    oracle_delta: torch.Tensor,
    max_shift: int = 6,
) -> tuple[int, int, float]:
    """Small diagnostic shift between model delta and oracle delta."""

    a = pred_delta.float().squeeze()
    b = oracle_delta.float().squeeze()
    if a.ndim != 2 or b.ndim != 2:
        return 0, 0, float("nan")

    best = (0, 0, float("-inf"))
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0a = max(0, dy)
            y1a = min(a.shape[0], b.shape[0] + dy)
            x0a = max(0, dx)
            x1a = min(a.shape[1], b.shape[1] + dx)
            y0b = max(0, -dy)
            y1b = y0b + (y1a - y0a)
            x0b = max(0, -dx)
            x1b = x0b + (x1a - x0a)
            if y1a <= y0a or x1a <= x0a:
                continue
            aa = a[y0a:y1a, x0a:x1a].flatten()
            bb = b[y0b:y1b, x0b:x1b].flatten()
            aa = aa - aa.mean()
            bb = bb - bb.mean()
            denom = torch.sqrt(aa.square().mean() * bb.square().mean()).item()
            score = float("nan") if denom < 1e-12 else float((aa * bb).mean().item() / denom)
            if math.isfinite(score) and score > best[2]:
                best = (dx, dy, score)
    return best


def metric_row(
    candidate: CropCandidate,
    model: str,
    pred_native: torch.Tensor,
    pred_large: torch.Tensor,
    oracle_native: torch.Tensor,
    oracle_large: torch.Tensor,
    multiplier: float,
    shave: int,
) -> Dict[str, object]:
    pred_n = valid_y(pred_native, shave)
    pred_l = valid_y(pred_large, shave)
    oracle_n = valid_y(oracle_native, shave)
    oracle_l = valid_y(oracle_large, shave)
    pred_delta = pred_l - pred_n
    oracle_delta = oracle_l - oracle_n
    tracking = pred_delta - oracle_delta
    oracle_err = pred_l - oracle_l
    shift_x, shift_y, shift_corr = best_delta_shift(pred_delta, oracle_delta)
    return {
        "dataset": candidate.dataset,
        "image": candidate.image,
        "x": candidate.x,
        "y": candidate.y,
        "crop_size": candidate.size,
        "texture_score": f"{candidate.texture_score:.8f}",
        "model": model,
        "cell_multiplier": f"{multiplier:g}",
        "oracle_rmse_y": f"{rmse(oracle_err):.8f}",
        "delta_tracking_rmse_y": f"{rmse(tracking):.8f}",
        "cell_sensitivity_rmse_y": f"{rmse(pred_delta):.8f}",
        "oracle_change_rmse_y": f"{rmse(oracle_delta):.8f}",
        "delta_oracle_corr_y": f"{corr(pred_delta, oracle_delta):.8f}",
        "delta_oracle_best_shift_x": shift_x,
        "delta_oracle_best_shift_y": shift_y,
        "delta_oracle_best_shift_corr_y": f"{shift_corr:.8f}",
        "highpass_delta_rms_y": f"{highpass_rms(pred_large - pred_native):.8f}",
    }


def score_candidate(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_model = {str(r["model"]): r for r in rows}
    sc = by_model.get("SC-INR")
    lte = by_model.get("LTE")
    nosinc = by_model.get("SC-INR-NoSinc")
    liif = by_model.get("LIIF")
    phasez = by_model.get("LTE-PhaseZ")

    def val(row: Dict[str, object] | None, key: str) -> float:
        if row is None:
            return float("nan")
        try:
            return float(row[key])
        except Exception:
            return float("nan")

    sc_track = val(sc, "delta_tracking_rmse_y")
    lte_track = val(lte, "delta_tracking_rmse_y")
    nosinc_track = val(nosinc, "delta_tracking_rmse_y")
    liif_track = val(liif, "delta_tracking_rmse_y")
    phasez_track = val(phasez, "delta_tracking_rmse_y")
    sc_oracle = val(sc, "oracle_rmse_y")
    lte_oracle = val(lte, "oracle_rmse_y")
    nosinc_oracle = val(nosinc, "oracle_rmse_y")
    sc_corr = val(sc, "delta_oracle_corr_y")
    lte_corr = val(lte, "delta_oracle_corr_y")
    nosinc_corr = val(nosinc, "delta_oracle_corr_y")
    oracle_change = val(sc, "oracle_change_rmse_y")

    adv_lte_track = lte_track - sc_track
    adv_nosinc_track = nosinc_track - sc_track
    adv_liif_track = liif_track - sc_track
    adv_lte_oracle = lte_oracle - sc_oracle
    adv_nosinc_oracle = nosinc_oracle - sc_oracle
    corr_gain_lte = sc_corr - lte_corr
    corr_gain_nosinc = sc_corr - nosinc_corr
    score_terms = [
        adv_lte_track,
        adv_nosinc_track,
        0.5 * adv_liif_track,
        0.5 * adv_lte_oracle,
        0.5 * adv_nosinc_oracle,
        0.02 * max(0.0, corr_gain_lte) if math.isfinite(corr_gain_lte) else float("nan"),
        0.02 * max(0.0, corr_gain_nosinc) if math.isfinite(corr_gain_nosinc) else float("nan"),
    ]
    score = sum(v for v in score_terms if math.isfinite(v))
    base = rows[0]
    return {
        "dataset": base["dataset"],
        "image": base["image"],
        "x": base["x"],
        "y": base["y"],
        "crop_size": base["crop_size"],
        "texture_score": base["texture_score"],
        "oracle_change_rmse_y": f"{oracle_change:.8f}",
        "sc_tracking_rmse_y": f"{sc_track:.8f}",
        "lte_tracking_rmse_y": f"{lte_track:.8f}",
        "nosinc_tracking_rmse_y": f"{nosinc_track:.8f}",
        "liif_tracking_rmse_y": f"{liif_track:.8f}",
        "phasez_tracking_rmse_y": f"{phasez_track:.8f}",
        "sc_oracle_rmse_y": f"{sc_oracle:.8f}",
        "lte_oracle_rmse_y": f"{lte_oracle:.8f}",
        "nosinc_oracle_rmse_y": f"{nosinc_oracle:.8f}",
        "sc_delta_oracle_corr_y": f"{sc_corr:.8f}",
        "lte_delta_oracle_corr_y": f"{lte_corr:.8f}",
        "nosinc_delta_oracle_corr_y": f"{nosinc_corr:.8f}",
        "adv_lte_tracking": f"{adv_lte_track:.8f}",
        "adv_nosinc_tracking": f"{adv_nosinc_track:.8f}",
        "adv_liif_tracking": f"{adv_liif_track:.8f}",
        "adv_phasez_tracking": f"{phasez_track - sc_track:.8f}",
        "adv_lte_oracle": f"{adv_lte_oracle:.8f}",
        "adv_nosinc_oracle": f"{adv_nosinc_oracle:.8f}",
        "selection_score": f"{score:.8f}",
        "passes_gate": int(adv_lte_track > 0 and adv_nosinc_track > 0 and adv_nosinc_oracle > 0),
    }


def diff_map(delta_y: torch.Tensor, vmax: float) -> np.ndarray:
    arr = delta_y.detach().float().squeeze().cpu().numpy()
    return np.clip(arr, -vmax, vmax)


def plot_candidate(
    out_path: Path,
    candidate: CropCandidate,
    images: Dict[str, Dict[str, torch.Tensor]],
    rows: Sequence[Dict[str, object]],
    multiplier: float,
) -> None:
    ensure_dir(out_path.parent)
    by_model = {str(r["model"]): r for r in rows}
    oracle_native = images["Oracle"]["native"]
    oracle_large = images["Oracle"]["large"]
    plot_models = [m for m in PLOT_MODELS if m in images]
    deltas = []
    for model in plot_models:
        native = images[model]["native"]
        large = images[model]["large"]
        deltas.append(valid_y(large, 0) - valid_y(native, 0))
    vmax = max(1e-5, float(torch.quantile(torch.cat([d.abs().flatten() for d in deltas]), 0.995).item()))

    nrows = len(plot_models)
    fig, axes = plt.subplots(nrows, 4, figsize=(12.2, 2.25 * nrows), dpi=180)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.patch.set_facecolor("#F7F8FA")
    for r, model in enumerate(plot_models):
        native = images[model]["native"]
        large = images[model]["large"]
        delta = valid_y(large, 0) - valid_y(native, 0)
        row = by_model.get(model)

        axes[r, 0].imshow(tensor_to_img(native))
        axes[r, 1].imshow(tensor_to_img(large))
        im = axes[r, 2].imshow(diff_map(delta, vmax), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[r, 3].imshow(tensor_to_img(oracle_large if model == "Oracle" else oracle_large))
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)
        if model == "Oracle":
            label = "HR box oracle"
        else:
            label = (
                f"{DISPLAY_NAMES[model]}\n"
                f"track={float(row['delta_tracking_rmse_y']):.4f}, "
                f"corr={float(row['delta_oracle_corr_y']):.2f}, "
                f"shift=({int(row['delta_oracle_best_shift_x'])},{int(row['delta_oracle_best_shift_y'])})"
            )
        axes[r, 0].set_ylabel(label, fontsize=9.0, rotation=0, ha="right", va="center", labelpad=58)
    axes[0, 0].set_title("native / reference", fontsize=10.5)
    axes[0, 1].set_title(f"cell x{multiplier:g}", fontsize=10.5)
    axes[0, 2].set_title(f"signed delta", fontsize=10.5)
    axes[0, 3].set_title("large-cell oracle", fontsize=10.5)
    fig.colorbar(im, ax=axes[:, 2], shrink=0.75, fraction=0.028, pad=0.010, label="Y signed change")
    fig.suptitle(
        f"Cell intervention: {candidate.dataset}/{candidate.image} crop(x={candidate.x}, y={candidate.y}, size={candidate.size})",
        fontsize=13.5,
        fontweight="bold",
        color="#20242A",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Lower tracking RMSE means the model's cell-induced change is closer to the HR box-average oracle change.",
        ha="center",
        fontsize=9.5,
        color="#5D6673",
    )
    fig.tight_layout(rect=(0.035, 0.035, 0.985, 0.965))
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_summary(out_path: Path, selected_rows: List[Dict[str, object]]) -> None:
    models = [m for m in DEFAULT_MODELS if any(str(r["model"]) == m for r in selected_rows)]
    metrics = ["delta_tracking_rmse_y", "oracle_rmse_y", "cell_sensitivity_rmse_y"]
    labels = ["Tracking RMSE\n(lower better)", "Oracle RMSE\n(lower better)", "Cell sensitivity"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), dpi=180)
    fig.patch.set_facecolor("#F7F8FA")
    for ax, metric, label in zip(axes, metrics, labels):
        means = []
        for model in models:
            vals = [float(r[metric]) for r in selected_rows if str(r["model"]) == model]
            means.append(float(np.mean(vals)))
        colors = [STYLE.get(m, "#888888") for m in models]
        ax.bar(models, means, color=colors, alpha=0.86)
        ax.set_title(label, fontsize=10.5)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Selected cell-intervention examples: averaged metrics", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    selected: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
) -> None:
    top = selected[0] if selected else {}
    pass_rate = float(np.mean([int(r["passes_gate"]) for r in summary_rows])) if summary_rows else float("nan")
    text = f"""# Cell Intervention Visualization

本目录用一个小型可视化实验展示 `SC-INR` 区别于 `LTE` 和 `SC-INR-NoSinc` 的核心行为。

## 协议

- 固定同一个 HR crop 生成 x{args.base_scale} LR 输入；
- 固定同一个 query grid；
- 只改变 decoder 输入的 cell multiplier：native `1` 和 large `{args.large_multiplier:g}`；
- 候选 crop 先按 GT 高频/局部纹理强度预筛，再按预定义 tracking 指标排序；
- 模型：`{','.join(parse_csv_list(args.models))}`。

## 主要产物

- `candidate_pool.csv`：GT 高频预筛候选 crop。
- `cell_intervention_metrics.csv`：逐 crop、逐模型指标。
- `selected_examples.csv`：按预定义分数排序后的候选。
- `figures/selected_*.png|pdf`：native cell、large cell、signed delta 与 oracle 对照图。
- `figures/selected_summary.png|pdf`：selected examples 上的平均指标柱状图。

## 当前最强展示例子

- top example: `{top.get('dataset', '')}/{top.get('image', '')}` crop `(x={top.get('x', '')}, y={top.get('y', '')}, size={top.get('crop_size', '')})`
- SC-INR tracking RMSE: `{top.get('sc_tracking_rmse_y', '')}`
- LTE tracking RMSE: `{top.get('lte_tracking_rmse_y', '')}`
- NoSinc tracking RMSE: `{top.get('nosinc_tracking_rmse_y', '')}`
- selection gate pass rate in candidate pool: `{pass_rate:.3f}`

## 如何解释

该实验最想展示的不是最终 PSNR，而是 cell path 的行为差异：

- `LTE`：cell 进入 learned phase，large-cell 改变可能表现为纹理相位/局部结构移动。
- `LTE-PhaseZ`：phase 来自 feature 而不是 cell，是去掉 cell-conditioned phase 的对照。
- `SC-INR-NoSinc`：去掉 analytic response 后，large-cell 输出几乎不变，说明它不是 footprint-aware。
- `SC-INR`：large-cell 产生非零响应，并且在 selected examples 上更接近 HR box-average oracle 的变化。

## 使用边界

这是 mechanism demonstration，不是 benchmark。它支持“SC-INR 的 cell path 行为区别于 LTE/NoSinc”，不能单独证明 `SC-INR` 全局视觉质量更好、sinc 是唯一因果，或最终 RGB 是 exact box integral。

## 复现命令

```bash
python scripts/analysis/visualize_cell_intervention.py \\
  --device {args.device} \\
  --datasets {args.datasets} \\
  --max_images {args.max_images} \\
  --crop_size {args.crop_size} \\
  --stride {args.stride} \\
  --max_crops_per_image {args.max_crops_per_image} \\
  --max_candidates {args.max_candidates} \\
  --num_select {args.num_select} \\
  --out {args.out}
```
"""
    (out_dir / "README_zh.md").write_text(text)


def run(args: argparse.Namespace) -> None:
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    candidates = enumerate_candidates(
        datasets,
        args.max_images,
        args.crop_size,
        args.stride,
        args.max_crops_per_image,
        args.base_scale,
    )
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    write_csv(
        out_dir / "candidate_pool.csv",
        [
            {
                "dataset": c.dataset,
                "image": c.image,
                "x": c.x,
                "y": c.y,
                "crop_size": c.size,
                "texture_score": f"{c.texture_score:.8f}",
            }
            for c in candidates
        ],
    )

    loaded = {name: load_model(name, device) for name in models}
    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    image_cache: Dict[tuple, Dict[str, Dict[str, torch.Tensor]]] = {}
    rows_cache: Dict[tuple, List[Dict[str, object]]] = {}

    try:
        for candidate in tqdm(candidates, desc="cell-intervention candidates"):
            lr, hr = make_lr_hr_crop(candidate, args.base_scale, device)
            oracle_native = hr
            oracle_large = box_oracle(hr, args.large_multiplier).clamp(0, 1)
            images: Dict[str, Dict[str, torch.Tensor]] = {
                "Oracle": {"native": oracle_native.detach().cpu(), "large": oracle_large.detach().cpu()}
            }
            rows: List[Dict[str, object]] = []
            shave = max(args.base_scale, int(math.ceil(args.large_multiplier / 2.0)) + 1)

            for model_name, model in loaded.items():
                pred_native = predict_with_cell(model, lr, 1.0, args.eval_bsize, args.base_scale)
                pred_large = predict_with_cell(model, lr, args.large_multiplier, args.eval_bsize, args.base_scale)
                row = metric_row(
                    candidate,
                    model_name,
                    pred_native,
                    pred_large,
                    oracle_native,
                    oracle_large,
                    args.large_multiplier,
                    shave,
                )
                rows.append(row)
                all_rows.append(row)
                images[model_name] = {
                    "native": pred_native.detach().cpu(),
                    "large": pred_large.detach().cpu(),
                }
                del pred_native, pred_large
            score_row = score_candidate(rows)
            summary_rows.append(score_row)
            key = (candidate.dataset, candidate.image, candidate.x, candidate.y, candidate.size)
            image_cache[key] = images
            rows_cache[key] = rows
            del lr, hr, oracle_large
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows.sort(key=lambda r: (int(r["passes_gate"]), float(r["selection_score"])), reverse=True)
    write_csv(out_dir / "cell_intervention_metrics.csv", all_rows)
    write_csv(out_dir / "selected_examples.csv", summary_rows)

    selected = summary_rows[: args.num_select]
    selected_long_rows: List[Dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        key = (str(row["dataset"]), str(row["image"]), int(row["x"]), int(row["y"]), int(row["crop_size"]))
        rows = rows_cache[key]
        images = image_cache[key]
        candidate = CropCandidate(
            dataset=str(row["dataset"]),
            image=str(row["image"]),
            path=DATASETS[str(row["dataset"])] / str(row["image"]),
            x=int(row["x"]),
            y=int(row["y"]),
            size=int(row["crop_size"]),
            texture_score=float(row["texture_score"]),
        )
        selected_long_rows.extend(rows)
        plot_candidate(fig_dir / f"selected_{rank:02d}_{candidate.dataset}_{Path(candidate.image).stem}.png", candidate, images, rows, args.large_multiplier)
    if selected_long_rows:
        plot_summary(fig_dir / "selected_summary.png", selected_long_rows)
    write_readme(out_dir, args, selected, summary_rows)
    print(f"Cell intervention visualization written to {out_dir}")
    if selected:
        print(json.dumps(selected[0], indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize cell-intervention behavior for SC-INR vs baselines.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", default="urban100,bsd100")
    parser.add_argument("--models", default="LIIF,LTE,LTE-PhaseZ,SC-INR,SC-INR-NoSinc")
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--base_scale", type=int, default=4)
    parser.add_argument("--large_multiplier", type=float, default=4.0)
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max_crops_per_image", type=int, default=4)
    parser.add_argument("--max_candidates", type=int, default=48)
    parser.add_argument("--num_select", type=int, default=4)
    parser.add_argument("--eval_bsize", type=int, default=50000)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "derived" / "diagnostics" / "cell_intervention_visual_2026-05-31",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args)


if __name__ == "__main__":
    main()
