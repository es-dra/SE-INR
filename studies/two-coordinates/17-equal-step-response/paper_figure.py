"""Render the complete equal-step grid from the frozen scorer's final summary.

This presentation script is post-processing, not part of the frozen hypothesis
or scoring pipeline. It never fits models, changes gates or selects images.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    completion = json.loads(args.summary.with_name("COMPLETE.json").read_text())
    digest = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    assert digest == completion["summary_sha256"]
    summary = json.loads(args.summary.read_text())
    rows = summary["scale_totals"]
    assert len(rows) == 48
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
                         "axes.labelsize": 8, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"DIV2K": "#3366a5", "Set14": "#d46745",
              "BSD100": "#489386", "Urban100": "#9a6cad"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True)
    for ax, step, letter in zip(axes, [-.25, .25, .5], "abc"):
        for dataset, color in colors.items():
            subset = sorted((r for r in rows if r["dataset"] == dataset
                             and r["h"] == step), key=lambda r: r["scale"])
            assert [r["scale"] for r in subset] == [2, 3, 4, 6]
            scale = np.array([r["scale"] for r in subset])
            point = np.array([r["J"] for r in subset])*1e6
            low = np.array([r["J_ci_low"] for r in subset])*1e6
            high = np.array([r["J_ci_high"] for r in subset])*1e6
            assert np.all(low > 0)
            ax.plot(scale, point, "o-", color=color, lw=1.2, ms=2.8, label=dataset)
            ax.fill_between(scale, low, high, color=color, alpha=.12, linewidth=0)
        ax.set_yscale("log")
        ax.set_xticks([2, 3, 4, 6])
        ax.set_xlabel("upsampling scale")
        ax.set_title(f"({letter}) equal step h = {step:+g}")
        ax.grid(axis="y", alpha=.17, linewidth=.5)
    axes[0].set_ylabel(r"$J$ ($10^{-6}$ MSE)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.53, -.015))
    fig.tight_layout(rect=(0, .12, 1, 1), pad=.6, w_pad=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"Subject": "Source summary SHA256 "+digest})
    plt.close(fig)


if __name__ == "__main__":
    main()
