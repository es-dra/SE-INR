#!/usr/bin/env python3
"""Property checks for the analytic sinc observation used by SC-INR.

This script verifies, on random 1D/2D Fourier components, that averaging a
Fourier signal over a finite box footprint matches the normalized sinc response
used in the SC-INR decoder. It is a theory-support diagnostic, not a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def analytic_average(
    kind: str,
    omega_h: float,
    omega_w: float,
    center_h: float,
    center_w: float,
    cell_h: float,
    cell_w: float,
    phase: float,
) -> float:
    center_arg = omega_h * center_h + omega_w * center_w + phase
    response = np.sinc(omega_h * cell_h / 2.0) * np.sinc(omega_w * cell_w / 2.0)
    if kind == "cos":
        return float(math.cos(math.pi * center_arg) * response)
    if kind == "sin":
        return float(math.sin(math.pi * center_arg) * response)
    raise ValueError(f"unsupported kind: {kind}")


def numeric_average(
    kind: str,
    omega_h: float,
    omega_w: float,
    center_h: float,
    center_w: float,
    cell_h: float,
    cell_w: float,
    phase: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    hh = center_h + 0.5 * cell_h * nodes
    ww = center_w + 0.5 * cell_w * nodes
    grid_h, grid_w = np.meshgrid(hh, ww, indexing="ij")
    arg = omega_h * grid_h + omega_w * grid_w + phase
    if kind == "cos":
        vals = np.cos(math.pi * arg)
    elif kind == "sin":
        vals = np.sin(math.pi * arg)
    else:
        raise ValueError(f"unsupported kind: {kind}")
    # Legendre weights integrate over [-1, 1]. After the box transform, the
    # area average is 1/4 * int_{-1}^{1} int_{-1}^{1} f(t, s) dt ds.
    return float(0.25 * np.einsum("i,j,ij->", weights, weights, vals))


def run_trials(args: argparse.Namespace) -> List[Dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    nodes, weights = np.polynomial.legendre.leggauss(args.quad_order)
    rows: List[Dict[str, object]] = []
    for trial in range(args.num_trials):
        omega_h = float(rng.uniform(-args.omega_bound, args.omega_bound))
        omega_w = float(rng.uniform(-args.omega_bound, args.omega_bound))
        center_h = float(rng.uniform(-1.0, 1.0))
        center_w = float(rng.uniform(-1.0, 1.0))
        phase = float(rng.uniform(-1.0, 1.0))
        # Match the feature-space cell range induced by scale factors around
        # x2.5 to x40; this covers the ID/OOD scales used in this project.
        scale_h = float(rng.uniform(args.scale_min, args.scale_max))
        scale_w = float(rng.uniform(args.scale_min, args.scale_max))
        cell_h = 2.0 / scale_h
        cell_w = 2.0 / scale_w
        for kind in ("cos", "sin"):
            ana = analytic_average(
                kind, omega_h, omega_w, center_h, center_w, cell_h, cell_w, phase
            )
            num = numeric_average(
                kind, omega_h, omega_w, center_h, center_w, cell_h, cell_w,
                phase, nodes, weights
            )
            rows.append({
                "trial": trial,
                "kind": kind,
                "omega_h": omega_h,
                "omega_w": omega_w,
                "center_h": center_h,
                "center_w": center_w,
                "phase": phase,
                "scale_h": scale_h,
                "scale_w": scale_w,
                "cell_h": cell_h,
                "cell_w": cell_w,
                "analytic_average": ana,
                "numeric_average": num,
                "abs_error": abs(ana - num),
                "rel_error": abs(ana - num) / max(1e-12, abs(num)),
            })
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace) -> Dict[str, object]:
    abs_err = np.array([float(r["abs_error"]) for r in rows])
    rel_err = np.array([float(r["rel_error"]) for r in rows])
    summary = {
        "num_trials": args.num_trials,
        "num_rows": len(rows),
        "seed": args.seed,
        "quad_order": args.quad_order,
        "omega_bound": args.omega_bound,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "abs_error_max": float(abs_err.max()),
        "abs_error_mean": float(abs_err.mean()),
        "abs_error_p99": float(np.quantile(abs_err, 0.99)),
        "rel_error_max": float(rel_err.max()),
        "rel_error_mean": float(rel_err.mean()),
        "rel_error_p99": float(np.quantile(rel_err, 0.99)),
    }
    path.write_text(json.dumps(summary, indent=2))
    return summary


def plot_errors(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    abs_err = np.array([float(r["abs_error"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(abs_err, bins=60, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Count")
    ax.set_title("Analytic sinc average vs. numerical box average")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "derived" / "analysis" / "sinc_theory_checks_2026-05-09" / "property_test")
    parser.add_argument("--num_trials", type=int, default=2000)
    parser.add_argument("--quad_order", type=int, default=64)
    parser.add_argument("--omega_bound", type=float, default=2.1)
    parser.add_argument("--scale_min", type=float, default=2.5)
    parser.add_argument("--scale_max", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)
    rows = run_trials(args)
    write_csv(out_dir / "sinc_property_trials.csv", rows)
    summary = write_summary(out_dir / "summary.json", rows, args)
    plot_errors(out_dir / "figures" / "abs_error_hist.png", rows)

    readme = (
        "# Sinc Observation Property Test\n\n"
        "This diagnostic compares the analytic normalized sinc response used by "
        "SC-INR with high-order numerical box integration on random 2D Fourier "
        "components.\n\n"
        f"- trials: `{args.num_trials}`\n"
        f"- quadrature order: `{args.quad_order}`\n"
        f"- max absolute error: `{summary['abs_error_max']:.6e}`\n"
        f"- p99 absolute error: `{summary['abs_error_p99']:.6e}`\n"
        f"- mean absolute error: `{summary['abs_error_mean']:.6e}`\n\n"
        "This supports the implementation-level observation formula only; it "
        "does not prove that sinc is the sole cause of benchmark gains.\n"
    )
    (out_dir / "README.md").write_text(readme)
    print(f"Sinc observation property test written to {out_dir}")


if __name__ == "__main__":
    main()
