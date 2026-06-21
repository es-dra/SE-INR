#!/usr/bin/env python3
"""Cell-to-phase proxy diagnostic for LTE-style decoders.

本脚本只检查机制路径：当 observation cell 改变时，模型内部是否存在
`cell -> phase` 的显式通路。它不跑图像重建，不证明视觉错误，只用于支撑
method 叙事中“cell-conditioned phase 允许尺度条件移动正弦基相位”的说法。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.evaluate_seed1_aux_metrics import load_model  # noqa: E402
from scripts.analysis.model_registry import STYLE  # noqa: E402


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def scale_label(x: float) -> str:
    return f"x{int(x)}" if float(x).is_integer() else f"x{x:g}"


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


def cell_phase_vector(model, rel_cell: torch.Tensor) -> torch.Tensor | None:
    phase = getattr(model, "phase", None)
    if phase is None or not isinstance(phase, torch.nn.Linear):
        return None
    with torch.no_grad():
        return phase(rel_cell)


def summarize_delta(vec: torch.Tensor | None, ref: torch.Tensor | None) -> Dict[str, float]:
    if vec is None or ref is None:
        return {
            "phase_l2": 0.0,
            "phase_abs_mean": 0.0,
            "phase_abs_max": 0.0,
            "delta_l2_vs_ref": 0.0,
            "delta_abs_mean_vs_ref": 0.0,
            "delta_abs_max_vs_ref": 0.0,
        }
    delta = vec - ref
    return {
        "phase_l2": float(torch.linalg.vector_norm(vec).item()),
        "phase_abs_mean": float(vec.abs().mean().item()),
        "phase_abs_max": float(vec.abs().max().item()),
        "delta_l2_vs_ref": float(torch.linalg.vector_norm(delta).item()),
        "delta_abs_mean_vs_ref": float(delta.abs().mean().item()),
        "delta_abs_max_vs_ref": float(delta.abs().max().item()),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, object]] = []
    scales = parse_floats(args.scales)
    multipliers = parse_floats(args.multipliers)

    for model_name in parse_csv_list(args.models):
        model = load_model(model_name, device)
        model.eval()
        has_cell_phase = cell_phase_vector(model, torch.zeros(1, 2, device=device)) is not None

        ref_rel = torch.tensor([[2.0 / float(args.ref_scale), 2.0 / float(args.ref_scale)]], device=device)
        ref_vec = cell_phase_vector(model, ref_rel)
        for scale in scales:
            rel = torch.tensor([[2.0 / scale, 2.0 / scale]], device=device)
            vec = cell_phase_vector(model, rel)
            rec: Dict[str, object] = {
                "model": model_name,
                "mode": "observation_scale",
                "scale_or_multiplier": scale_label(scale),
                "ref": scale_label(float(args.ref_scale)),
                "rel_cell_x": float(rel[0, 0].item()),
                "rel_cell_y": float(rel[0, 1].item()),
                "has_cell_phase": int(has_cell_phase),
            }
            rec.update(summarize_delta(vec, ref_vec))
            rows.append(rec)

        native_rel = torch.tensor([[2.0 / float(args.base_scale), 2.0 / float(args.base_scale)]], device=device)
        native_vec = cell_phase_vector(model, native_rel)
        for multiplier in multipliers:
            rel = torch.tensor(
                [[2.0 * multiplier / float(args.base_scale), 2.0 * multiplier / float(args.base_scale)]],
                device=device,
            )
            vec = cell_phase_vector(model, rel)
            rec = {
                "model": model_name,
                "mode": "cell_multiplier",
                "scale_or_multiplier": f"{multiplier:g}",
                "ref": "1",
                "rel_cell_x": float(rel[0, 0].item()),
                "rel_cell_y": float(rel[0, 1].item()),
                "has_cell_phase": int(has_cell_phase),
            }
            rec.update(summarize_delta(vec, native_vec))
            rows.append(rec)
        del model

    write_csv(out_dir / "cell_phase_proxy.csv", rows)
    plot(rows, out_dir / "figures")
    write_readme(out_dir, args, rows)
    print(f"Cell-phase proxy diagnostic written to {out_dir}")


def plot(rows: Sequence[Dict[str, object]], fig_dir: Path) -> None:
    ensure_dir(fig_dir)
    for mode, xlabel, filename in [
        ("observation_scale", "Observation scale", "phase_delta_by_scale.png"),
        ("cell_multiplier", "Cell multiplier", "phase_delta_by_multiplier.png"),
    ]:
        sub = [r for r in rows if r["mode"] == mode]
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
        for model in sorted({str(r["model"]) for r in sub}):
            mr = [r for r in sub if str(r["model"]) == model]
            xs = []
            ys = []
            for r in mr:
                label = str(r["scale_or_multiplier"])
                xs.append(float(label[1:] if label.startswith("x") else label))
                ys.append(float(r["delta_abs_mean_vs_ref"]))
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            ax.plot(xs, ys, marker="o", color=STYLE.get(model, "#888888"), label=model)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean |phase(c)-phase(ref)|")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=220)
        fig.savefig((fig_dir / filename).with_suffix(".pdf"))
        plt.close(fig)


def write_readme(out_dir: Path, args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> None:
    lte_rows = [r for r in rows if r["model"] == "LTE" and r["mode"] == "observation_scale"]
    max_lte = max([float(r["delta_abs_mean_vs_ref"]) for r in lte_rows], default=float("nan"))
    text = f"""# Cell-Phase Proxy Diagnostic

本目录检查 `cell -> phase` 的显式机制路径。

## 协议

- 模型：`{args.models}`
- observation scales：`{args.scales}`，参考尺度 `x{args.ref_scale:g}`
- cell multipliers：`{args.multipliers}`，参考 multiplier `1`

## 主要结论读法

- `LTE` 有 `phase = Linear(cell)`，因此 `cell` 改变会直接改变 Fourier-like basis phase。
- `LTE-PhaseZ`、`SC-INR`、`SC-INR-NoSinc` 没有 cell-conditioned phase，本诊断中该项应为 0。
- 本诊断只说明“是否存在移动 phase 的机制能力”，不证明一定产生视觉错误。

当前 LTE 最大 mean phase delta vs ref：`{max_lte:.6f}`。

## 文件

- `cell_phase_proxy.csv`
- `figures/phase_delta_by_scale.png|pdf`
- `figures/phase_delta_by_multiplier.png|pdf`
- `run_config.json`
"""
    (out_dir / "README_zh.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze explicit cell-conditioned phase paths.")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "derived" / "diagnostics" / "cell_phase_proxy_2026-06-20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="LTE,LTE-PhaseZ,SC-INR,SC-INR-NoSinc")
    parser.add_argument("--scales", default="2,3,4,6,8,12,16,24,30")
    parser.add_argument("--ref_scale", type=float, default=4.0)
    parser.add_argument("--base_scale", type=float, default=4.0)
    parser.add_argument("--multipliers", default="0.5,1,2,4")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
