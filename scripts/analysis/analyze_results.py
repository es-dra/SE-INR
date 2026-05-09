#!/usr/bin/env python3
"""Reusable result aggregation and plotting for SC-INR experiments.

This script is intentionally read-only with respect to model/checkpoint files. It
summarizes the current single-seed result files and, when available, additional
multi-seed benchmark files under results/seeds/.

Outputs are written to results/analysis/ by default:
  - benchmark_summary.csv / benchmark_summary.tex
  - benchmark_delta_vs_lte.csv
  - benchmark_seed_mean_std.csv / benchmark_seed_mean_std.tex
  - figures/benchmark_ood_avg.png
  - figures/benchmark_delta_vs_lte.png
  - figures/continuous_<dataset>.png
  - figures/fce_bar.png, if fce.json has data
  - figures/phase_intervention_ood.png, if phase_intervention.json has data
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ID_SCALES = ["x2", "x3", "x4"]
OOD_SCALES = ["x6", "x8", "x12", "x16", "x24", "x30"]
BENCHMARKS = ["Set5", "Set14", "BSD100", "Urban100"]
MODEL_ALIASES = {
    "LTE-NoCell": "LTE-NoCellPhase",
    "LTE-NoC": "LTE-NoCellPhase",
    "LTE-FeaturePhase": "LTE-PhaseZ",
    "SC-INR-Adaptive": "SC-INR-NoPhi",
    "SC-INR-Fixed": "SC-INR-FixedOmega",
    "SC-INR+PhiZ": "SC-INR",
    "SC-INR-Signed": "SC-INR-NoPhi-Signed",
    "SC-INR-Adaptive-Signed": "SC-INR-NoPhi-Signed",
}
CORE_MODELS = ["LIIF", "LTE", "SC-INR-NoPhi"]
MODEL_ORDER = [
    "LIIF", "LIIF-EQ", "LTE", "LTE-EQ", "LTE-NoCellPhase", "LTE-PhaseZ",
    "SC-INR-FixedOmega", "SC-INR-NoPhi", "SC-INR-NoPhi-Signed", "SC-INR",
]
STYLE = {
    "LIIF": {"color": "#4C72B0", "ls": "--", "lw": 1.8},
    "LIIF-EQ": {"color": "#4C72B0", "ls": "-", "lw": 1.5},
    "LTE": {"color": "#DD8452", "ls": "--", "lw": 1.8},
    "LTE-EQ": {"color": "#DD8452", "ls": "-", "lw": 1.5},
    "LTE-NoCellPhase": {"color": "#55A868", "ls": ":", "lw": 1.6},
    "LTE-PhaseZ": {"color": "#8172B2", "ls": ":", "lw": 1.6},
    "SC-INR-FixedOmega": {"color": "#C44E52", "ls": "--", "lw": 1.8},
    "SC-INR-NoPhi": {"color": "#C44E52", "ls": "-", "lw": 2.0},
    "SC-INR-NoPhi-Signed": {"color": "#8B0000", "ls": "-", "lw": 2.0},
    "SC-INR": {"color": "#222222", "ls": "-", "lw": 2.4},
}
LEGACY_SCINR_NOPHI_FILES = {
    "benchmark.json",
    "benchmark_seed2.json",
    "benchmark_seed3.json",
}


def normalize_model_names(data: Dict[str, Any], source_path: Path | None = None) -> Dict[str, Any]:
    # Older result files used raw key "SC-INR" for the no-phase adaptive
    # variant. New paper-facing files reserve "SC-INR" for the PhiZ final
    # candidate, so only remap the ambiguous key in known legacy contexts.
    source_name = source_path.name if source_path is not None else ""
    legacy_sc_inr_no_phi = (
        "SC-INR+PhiZ" in data
        or source_name in LEGACY_SCINR_NOPHI_FILES
    )
    normalized: Dict[str, Any] = {}
    for model, value in data.items():
        if model == "SC-INR" and legacy_sc_inr_no_phi:
            canonical = "SC-INR-NoPhi"
        else:
            canonical = MODEL_ALIASES.get(model, model)
        normalized[canonical] = value
    return normalized


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    return normalize_model_names(data, path)


def seed1_benchmark_path(root: Path) -> Path:
    enriched = root / "results" / "benchmark_seed1_with_signed_phiz.json"
    if enriched.exists():
        return enriched
    return root / "results" / "benchmark.json"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def numeric_values_for_scales(model_data: Dict[str, Any], scales: Iterable[str]) -> List[float]:
    vals: List[float] = []
    for dataset in BENCHMARKS:
        d = model_data.get(dataset, {})
        for scale in scales:
            v = d.get(scale)
            if isinstance(v, (int, float)) and not math.isnan(v):
                vals.append(float(v))
    return vals


def average_or_nan(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def ordered_models(data: Dict[str, Any]) -> List[str]:
    seen = [m for m in MODEL_ORDER if m in data]
    seen += [m for m in data if m not in seen]
    return seen


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_simple_latex_table(path: Path, headers: List[str], rows: List[List[str]], caption: str, label: str) -> None:
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + " " + chr(92) * 2,
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " " + chr(92) * 2)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines))


def summarize_benchmark(root: Path, out_dir: Path) -> Dict[str, Any]:
    data = load_json(seed1_benchmark_path(root))
    rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    for model in ordered_models(data):
        md = data.get(model, {})
        id_avg = average_or_nan(numeric_values_for_scales(md, ID_SCALES))
        ood_avg = average_or_nan(numeric_values_for_scales(md, OOD_SCALES))
        all_avg = average_or_nan(numeric_values_for_scales(md, ID_SCALES + OOD_SCALES))
        rows.append({"model": model, "id_avg": f"{id_avg:.4f}", "ood_avg": f"{ood_avg:.4f}", "all_avg": f"{all_avg:.4f}"})

    lte = data.get("LTE", {})
    for model in ordered_models(data):
        if model == "LTE":
            continue
        md = data.get(model, {})
        for dataset in BENCHMARKS:
            for scale in ID_SCALES + OOD_SCALES:
                v = md.get(dataset, {}).get(scale)
                b = lte.get(dataset, {}).get(scale)
                if isinstance(v, (int, float)) and isinstance(b, (int, float)):
                    delta_rows.append({
                        "model": model,
                        "dataset": dataset,
                        "scale": scale,
                        "delta_vs_lte": f"{float(v) - float(b):.4f}",
                    })

    write_csv(out_dir / "benchmark_summary.csv", rows, ["model", "id_avg", "ood_avg", "all_avg"])
    write_csv(out_dir / "benchmark_delta_vs_lte.csv", delta_rows, ["model", "dataset", "scale", "delta_vs_lte"])
    write_simple_latex_table(
        out_dir / "benchmark_summary.tex",
        ["Model", "ID Avg.", "OOD Avg.", "All Avg."],
        [[r["model"], r["id_avg"], r["ood_avg"], r["all_avg"]] for r in rows],
        "Single-seed benchmark summary computed from results/benchmark.json. ID uses x2/x3/x4 and OOD uses x6/x8/x12/x16/x24/x30 across four datasets.",
        "tab:single_seed_benchmark_summary",
    )

    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    labels = [r["model"] for r in rows]
    ood = [float(r["ood_avg"]) for r in rows]
    colors = [STYLE.get(m, {}).get("color", "#888888") for m in labels]
    plt.figure(figsize=(10, 4.5))
    plt.bar(labels, ood, color=colors)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("OOD PSNR Avg. (dB)")
    plt.title("Single-seed OOD benchmark average")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "benchmark_ood_avg.png", dpi=180)
    plt.close()

    # Aggregate model-level OOD delta against LTE.
    delta_by_model: Dict[str, List[float]] = {}
    for r in delta_rows:
        if r["scale"] in OOD_SCALES:
            delta_by_model.setdefault(r["model"], []).append(float(r["delta_vs_lte"]))
    dlabels = [m for m in ordered_models(data) if m in delta_by_model]
    dvals = [float(np.mean(delta_by_model[m])) for m in dlabels]
    plt.figure(figsize=(10, 4.5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.bar(dlabels, dvals, color=[STYLE.get(m, {}).get("color", "#888888") for m in dlabels])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("OOD PSNR delta vs. LTE (dB)")
    plt.title("Single-seed OOD gain relative to LTE")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "benchmark_delta_vs_lte.png", dpi=180)
    plt.close()
    return data


def load_seed_benchmarks(root: Path) -> Dict[int, Dict[str, Any]]:
    seed_data: Dict[int, Dict[str, Any]] = {1: load_json(seed1_benchmark_path(root))}
    seeds_dir = root / "results" / "seeds"
    for path in sorted(seeds_dir.glob("benchmark_seed*.json")):
        suffix = path.stem.replace("benchmark_seed", "")
        if suffix.isdigit():
            seed_data[int(suffix)] = load_json(path)
    return {k: v for k, v in seed_data.items() if v}


def summarize_seeds(root: Path, out_dir: Path) -> None:
    seed_data = load_seed_benchmarks(root)
    rows: List[Dict[str, Any]] = []
    all_models = sorted(set().union(*(d.keys() for d in seed_data.values())), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999)
    for model in all_models:
        id_vals, ood_vals, seeds_present = [], [], []
        for seed, data in sorted(seed_data.items()):
            if model not in data:
                continue
            seeds_present.append(seed)
            id_vals.append(average_or_nan(numeric_values_for_scales(data[model], ID_SCALES)))
            ood_vals.append(average_or_nan(numeric_values_for_scales(data[model], OOD_SCALES)))
        id_clean = [v for v in id_vals if not math.isnan(v)]
        ood_clean = [v for v in ood_vals if not math.isnan(v)]
        if not id_clean and not ood_clean:
            continue
        rows.append({
            "model": model,
            "seeds": ";".join(map(str, seeds_present)),
            "n": len(seeds_present),
            "id_mean": f"{np.mean(id_clean):.4f}" if id_clean else "nan",
            "id_std": f"{np.std(id_clean, ddof=1):.4f}" if len(id_clean) > 1 else "nan",
            "ood_mean": f"{np.mean(ood_clean):.4f}" if ood_clean else "nan",
            "ood_std": f"{np.std(ood_clean, ddof=1):.4f}" if len(ood_clean) > 1 else "nan",
        })
    write_csv(out_dir / "benchmark_seed_mean_std.csv", rows, ["model", "seeds", "n", "id_mean", "id_std", "ood_mean", "ood_std"])
    latex_rows = []
    for r in rows:
        id_cell = r["id_mean"] if r["id_std"] == "nan" else f"{r['id_mean']} $\\pm$ {r['id_std']}"
        ood_cell = r["ood_mean"] if r["ood_std"] == "nan" else f"{r['ood_mean']} $\\pm$ {r['ood_std']}"
        latex_rows.append([r["model"], str(r["n"]), id_cell, ood_cell])
    write_simple_latex_table(
        out_dir / "benchmark_seed_mean_std.tex",
        ["Model", "Seeds", "ID Avg.", "OOD Avg."],
        latex_rows,
        "Benchmark mean and standard deviation across available seeds. Seed 1 uses benchmark_seed1_with_signed_phiz.json when present; additional seeds are loaded from results/seeds/benchmark_seed*.json.",
        "tab:benchmark_seed_mean_std",
    )


def plot_continuous(root: Path, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    cont_dir = root / "results" / "continuous"
    for path in sorted(cont_dir.glob("*.json")):
        data = load_json(path)
        plt.figure(figsize=(10, 4.5))
        for model in ordered_models(data):
            md = data[model]
            xs, ys = [], []
            for k, v in md.items():
                if v is None:
                    continue
                try:
                    s = float(str(k).lstrip("x"))
                except ValueError:
                    continue
                if s < 1.1:
                    continue
                xs.append(s)
                ys.append(float(v))
            if not xs:
                continue
            order = np.argsort(xs)
            style = STYLE.get(model, {"color": "#888888", "ls": "-", "lw": 1.4})
            plt.plot(np.array(xs)[order], np.array(ys)[order], label=model, color=style["color"], linestyle=style["ls"], linewidth=style["lw"])
        plt.axvspan(1.1, 4.0, color="#4C72B0", alpha=0.06, label="train scale range")
        plt.axvline(4.0, color="gray", linestyle=":", linewidth=1)
        plt.xlabel("Scale")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Continuous-scale PSNR: {path.stem}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"continuous_{path.stem}.png", dpi=180)
        plt.close()


def plot_fce(root: Path, out_dir: Path) -> None:
    data = load_json(root / "results" / "fce.json")
    if not data:
        return
    rows = []
    for model, vals in data.items():
        numeric = [float(v) for v in vals.values() if isinstance(v, (int, float))]
        if numeric:
            rows.append({"model": model, "fce_mean": f"{np.mean(numeric):.6g}", "n": len(numeric)})
    if not rows:
        return
    write_csv(out_dir / "fce_summary.csv", rows, ["model", "fce_mean", "n"])
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    plt.figure(figsize=(7, 4))
    plt.bar([r["model"] for r in rows], [float(r["fce_mean"]) for r in rows], color="#55A868")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("FCE mean")
    plt.title("Function Consistency Error summary")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "fce_bar.png", dpi=180)
    plt.close()


def plot_phase_intervention(root: Path, out_dir: Path) -> None:
    data = load_json(root / "results" / "phase_intervention.json")
    if not data:
        return
    rows = []
    for model, flat in data.items():
        id_vals, ood_vals = [], []
        for key, v in flat.items():
            if not isinstance(v, (int, float)):
                continue
            scale = "x" + key.split("_x")[-1]
            if scale in ID_SCALES:
                id_vals.append(float(v))
            elif scale in OOD_SCALES:
                ood_vals.append(float(v))
        rows.append({
            "model": model,
            "id_avg": f"{np.mean(id_vals):.4f}" if id_vals else "nan",
            "ood_avg": f"{np.mean(ood_vals):.4f}" if ood_vals else "nan",
        })
    write_csv(out_dir / "phase_intervention_summary.csv", rows, ["model", "id_avg", "ood_avg"])
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    plt.figure(figsize=(7, 4))
    plt.bar([r["model"] for r in rows], [float(r["ood_avg"]) for r in rows], color="#DD8452")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("OOD PSNR Avg. (dB)")
    plt.title("Phase intervention OOD summary")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "phase_intervention_ood.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."), help="Equivariant-ASISR project root")
    parser.add_argument("--out", type=Path, default=Path("results/analysis"), help="Output directory relative to root unless absolute")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out if args.out.is_absolute() else root / args.out
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    summarize_benchmark(root, out_dir)
    summarize_seeds(root, out_dir)
    plot_continuous(root, out_dir)
    plot_fce(root, out_dir)
    plot_phase_intervention(root, out_dir)
    print(f"Analysis outputs written to {out_dir}")


if __name__ == "__main__":
    main()
