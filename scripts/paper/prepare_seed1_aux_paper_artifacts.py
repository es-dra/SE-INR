#!/usr/bin/env python3
"""Prepare paper-facing tables and figures from seed1 auxiliary metrics.

This script is read-only with respect to checkpoints and raw experiment CSVs.
It consumes the outputs of scripts/evaluate_seed1_aux_metrics.py and writes a
paper-oriented summary under:

  results/analysis/seed1_aux_metrics_all8/paper/

The intended use is to keep the original metric files intact while generating
compact tables/plots for the SC-INR paper narrative.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "results" / "analysis" / "seed1_aux_metrics_all8"
DEFAULT_OUT = DEFAULT_IN / "paper"

MODEL_ORDER = [
    "LIIF",
    "LIIF-EQ",
    "LTE",
    "LTE-EQ",
    "LTE-NoCell",
    "LTE-FeaturePhase",
    "SC-INR-Fixed",
    "SC-INR",
]

MODEL_ALIASES = {
    "SC-INR-Adaptive": "SC-INR",
    "SC-INR-Adaptive-Signed": "SC-INR-Signed",
}

DISPLAY_DATASET = {
    "bsd100": "BSD100",
    "urban100": "Urban100",
}

STYLE = {
    "LIIF": "#4C72B0",
    "LIIF-EQ": "#64B5CD",
    "LTE": "#DD8452",
    "LTE-EQ": "#DDAA33",
    "LTE-NoCell": "#55A868",
    "LTE-FeaturePhase": "#8172B2",
    "SC-INR-Fixed": "#C44E52",
    "SC-INR": "#8B0000",
    "SC-INR-Signed": "#AA3377",
}

QUALITY_METRICS = [
    "psnr_y",
    "ssim_y",
    "rmse_y",
    "edge_rmse",
    "texture_rmse",
    "flat_rmse",
    "highpass_rmse",
]

CONSISTENCY_METRICS = [
    "consistency_psnr_y",
    "consistency_rmse_y",
    "edge_consistency_rmse",
    "texture_consistency_rmse",
    "flat_consistency_rmse",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def order_models(models: Iterable[str]) -> List[str]:
    seen = [m for m in MODEL_ORDER if m in set(models)]
    seen += [m for m in models if m not in seen]
    return seen


def normalize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model" in df.columns:
        df["model"] = df["model"].replace(MODEL_ALIASES)
    return df


def scale_to_int(scale: str) -> int:
    return int(str(scale).lower().replace("x", ""))


def add_lte_deltas(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    baseline = df[df["model"] == "LTE"][group_cols + metrics].copy()
    baseline = baseline.rename(columns={m: f"{m}_lte" for m in metrics})
    out = df.merge(baseline, on=group_cols, how="left")
    for metric in metrics:
        out[f"delta_{metric}_vs_lte"] = out[metric] - out[f"{metric}_lte"]
    return out


def latex_escape(text: object) -> str:
    return str(text).replace("_", "\\_")


def write_latex_table(path: Path, headers: List[str], rows: List[List[str]], caption: str, label: str,
                      table_star: bool = False) -> None:
    env = "table*" if table_star else "table"
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    path.write_text("\n".join(lines))


def fmt_psnr(x: float) -> str:
    return f"{x:.3f}"


def fmt_psnr_delta(x: float) -> str:
    return f"{x:+.3f}"


def fmt_ssim(x: float) -> str:
    return f"{x:.4f}"


def fmt_ssim_delta(x: float) -> str:
    return f"{x:+.4f}"


def fmt_rmse(x: float) -> str:
    return f"{x:.5f}"


def fmt_rmse_delta(x: float) -> str:
    return f"{x:+.5f}"


def load_quality(in_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality = pd.read_csv(in_dir / "quality_summary.csv")
    quality = normalize_model_names(quality)
    quality = quality[quality["model"] != "Bicubic"].copy()
    quality["scale_num"] = quality["scale"].map(scale_to_int)
    quality["split"] = np.where(quality["scale_num"] == 4, "x4", "x8-x30")

    quality_per_scale = add_lte_deltas(
        quality,
        ["dataset", "scale"],
        QUALITY_METRICS,
    )

    quality_split = (
        quality
        .groupby(["model", "dataset", "split"], as_index=False)[QUALITY_METRICS]
        .mean()
    )
    quality_split = add_lte_deltas(
        quality_split,
        ["dataset", "split"],
        QUALITY_METRICS,
    )
    quality_split["model"] = pd.Categorical(quality_split["model"], MODEL_ORDER, ordered=True)
    quality_split = quality_split.sort_values(["model", "dataset", "split"]).reset_index(drop=True)
    quality_per_scale["model"] = pd.Categorical(quality_per_scale["model"], MODEL_ORDER, ordered=True)
    quality_per_scale = quality_per_scale.sort_values(["model", "dataset", "scale_num"]).reset_index(drop=True)
    return quality, quality_per_scale, quality_split


def load_consistency(in_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    consistency = pd.read_csv(in_dir / "consistency_summary.csv")
    consistency = normalize_model_names(consistency)
    consistency["source_scale_num"] = consistency["source_scale"].map(scale_to_int)
    consistency_agg = (
        consistency
        .groupby(["model", "dataset"], as_index=False)[CONSISTENCY_METRICS]
        .mean()
    )
    consistency_agg = add_lte_deltas(consistency_agg, ["dataset"], CONSISTENCY_METRICS)
    consistency = add_lte_deltas(
        consistency,
        ["dataset", "source_scale", "observation_scale"],
        CONSISTENCY_METRICS,
    )
    consistency_agg["model"] = pd.Categorical(consistency_agg["model"], MODEL_ORDER, ordered=True)
    consistency_agg = consistency_agg.sort_values(["model", "dataset"]).reset_index(drop=True)
    return consistency, consistency_agg


def lookup(df: pd.DataFrame, model: str, dataset: str, split: str | None = None) -> pd.Series:
    mask = (df["model"].astype(str) == model) & (df["dataset"] == dataset)
    if split is not None:
        mask &= df["split"] == split
    rows = df[mask]
    if rows.empty:
        raise KeyError((model, dataset, split))
    return rows.iloc[0]


def write_quality_tables(out_dir: Path, quality_split: pd.DataFrame) -> None:
    rows = []
    for model in order_models(quality_split["model"].astype(str).unique()):
        bsd_id = lookup(quality_split, model, "bsd100", "x4")
        bsd_ood = lookup(quality_split, model, "bsd100", "x8-x30")
        urb_id = lookup(quality_split, model, "urban100", "x4")
        urb_ood = lookup(quality_split, model, "urban100", "x8-x30")
        rows.append([
            latex_escape(model),
            fmt_psnr(bsd_id["psnr_y"]),
            fmt_psnr(bsd_ood["psnr_y"]),
            fmt_psnr_delta(bsd_ood["delta_psnr_y_vs_lte"]),
            fmt_psnr(urb_id["psnr_y"]),
            fmt_psnr(urb_ood["psnr_y"]),
            fmt_psnr_delta(urb_ood["delta_psnr_y_vs_lte"]),
        ])
    write_latex_table(
        out_dir / "quality_psnr_table.tex",
        [
            "Model",
            "BSD100 x4",
            "BSD100 x8--x30",
            "$\\Delta$",
            "Urban100 x4",
            "Urban100 x8--x30",
            "$\\Delta$",
        ],
        rows,
        "Seed-1 auxiliary quality summary. OOD averages are equally averaged over x8, x16, and x30. Deltas are against LTE under the same HR-downsampled protocol.",
        "tab:seed1_aux_quality_psnr",
        table_star=True,
    )

    rows = []
    for model in order_models(quality_split["model"].astype(str).unique()):
        bsd_ood = lookup(quality_split, model, "bsd100", "x8-x30")
        urb_ood = lookup(quality_split, model, "urban100", "x8-x30")
        rows.append([
            latex_escape(model),
            fmt_ssim(bsd_ood["ssim_y"]),
            fmt_ssim_delta(bsd_ood["delta_ssim_y_vs_lte"]),
            fmt_rmse(bsd_ood["texture_rmse"]),
            fmt_rmse_delta(bsd_ood["delta_texture_rmse_vs_lte"]),
            fmt_ssim(urb_ood["ssim_y"]),
            fmt_ssim_delta(urb_ood["delta_ssim_y_vs_lte"]),
            fmt_rmse(urb_ood["texture_rmse"]),
            fmt_rmse_delta(urb_ood["delta_texture_rmse_vs_lte"]),
        ])
    write_latex_table(
        out_dir / "quality_ssim_texture_table.tex",
        [
            "Model",
            "BSD SSIM",
            "$\\Delta$",
            "BSD tex RMSE",
            "$\\Delta$",
            "Urban SSIM",
            "$\\Delta$",
            "Urban tex RMSE",
            "$\\Delta$",
        ],
        rows,
        "Seed-1 OOD SSIM-Y and texture-region RMSE. Texture regions are the top 20 percent local-variance pixels. Lower RMSE is better.",
        "tab:seed1_aux_quality_texture",
        table_star=True,
    )


def write_consistency_table(out_dir: Path, consistency_agg: pd.DataFrame) -> None:
    rows = []
    for model in order_models(consistency_agg["model"].astype(str).unique()):
        bsd = consistency_agg[(consistency_agg["model"].astype(str) == model) & (consistency_agg["dataset"] == "bsd100")].iloc[0]
        urb = consistency_agg[(consistency_agg["model"].astype(str) == model) & (consistency_agg["dataset"] == "urban100")].iloc[0]
        rows.append([
            latex_escape(model),
            f"{bsd['consistency_psnr_y']:.2f}",
            f"{bsd['delta_consistency_psnr_y_vs_lte']:+.2f}",
            fmt_rmse(bsd["texture_consistency_rmse"]),
            fmt_rmse_delta(bsd["delta_texture_consistency_rmse_vs_lte"]),
            f"{urb['consistency_psnr_y']:.2f}",
            f"{urb['delta_consistency_psnr_y_vs_lte']:+.2f}",
            fmt_rmse(urb["texture_consistency_rmse"]),
            fmt_rmse_delta(urb["delta_texture_consistency_rmse_vs_lte"]),
        ])
    write_latex_table(
        out_dir / "consistency_table.tex",
        [
            "Model",
            "BSD SC-PSNR",
            "$\\Delta$",
            "BSD tex RMSE",
            "$\\Delta$",
            "Urban SC-PSNR",
            "$\\Delta$",
            "Urban tex RMSE",
            "$\\Delta$",
        ],
        rows,
        "Seed-1 same-LR cross-scale observation consistency, averaged over x8/x16/x30 to x4. LTE-NoCell and LTE-FeaturePhase are diagnostic baselines with weakened cell response, so their high consistency should be interpreted with reconstruction quality.",
        "tab:seed1_aux_consistency",
        table_star=True,
    )


def savefig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_scale_gain(out_dir: Path, quality_per_scale: pd.DataFrame) -> None:
    plot_models = ["LIIF", "LIIF-EQ", "LTE-EQ", "LTE-NoCell", "LTE-FeaturePhase", "SC-INR-Fixed", "SC-INR"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for ax, dataset in zip(axes, ["bsd100", "urban100"]):
        for model in plot_models:
            sub = quality_per_scale[
                (quality_per_scale["model"].astype(str) == model)
                & (quality_per_scale["dataset"] == dataset)
            ].sort_values("scale_num")
            if sub.empty:
                continue
            ax.plot(
                sub["scale_num"],
                sub["delta_psnr_y_vs_lte"],
                marker="o",
                linewidth=2.0 if model == "SC-INR" else 1.3,
                color=STYLE.get(model, "#888888"),
                label=model,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(DISPLAY_DATASET[dataset])
        ax.set_xlabel("Scale")
        ax.set_xticks([4, 8, 16, 30])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("PSNR-Y gain vs. LTE (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(top=0.78)
    savefig(fig, out_dir / "fig_scale_gain_vs_lte")


def plot_ood_quality(out_dir: Path, quality_split: pd.DataFrame) -> None:
    ood = quality_split[quality_split["split"] == "x8-x30"].copy()
    models = order_models(ood["model"].astype(str).unique())
    x = np.arange(len(models))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for ax, dataset in zip(axes, ["bsd100", "urban100"]):
        sub = ood[ood["dataset"] == dataset]
        vals = []
        for model in models:
            m = sub[sub["model"].astype(str) == model]
            vals.append(float(m["delta_psnr_y_vs_lte"].iloc[0]) if not m.empty else np.nan)
        ax.bar(x, vals, width=0.68, color=[STYLE.get(m, "#888888") for m in models])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(DISPLAY_DATASET[dataset])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("OOD PSNR-Y gain vs. LTE (dB)")
    savefig(fig, out_dir / "fig_ood_psnr_gain_vs_lte")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for ax, dataset in zip(axes, ["bsd100", "urban100"]):
        sub = ood[ood["dataset"] == dataset]
        vals = []
        for model in models:
            m = sub[sub["model"].astype(str) == model]
            vals.append(float(m["delta_texture_rmse_vs_lte"].iloc[0]) if not m.empty else np.nan)
        ax.bar(x, vals, width=0.68, color=[STYLE.get(m, "#888888") for m in models])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(DISPLAY_DATASET[dataset])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("OOD texture RMSE delta vs. LTE")
    savefig(fig, out_dir / "fig_ood_texture_rmse_delta_vs_lte")


def plot_consistency_tradeoff(out_dir: Path, quality_split: pd.DataFrame, consistency_agg: pd.DataFrame) -> None:
    ood = quality_split[quality_split["split"] == "x8-x30"][
        ["model", "dataset", "delta_psnr_y_vs_lte"]
    ]
    trade = ood.merge(
        consistency_agg[["model", "dataset", "delta_consistency_psnr_y_vs_lte"]],
        on=["model", "dataset"],
        how="inner",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, dataset in zip(axes, ["bsd100", "urban100"]):
        sub = trade[trade["dataset"] == dataset]
        for _, row in sub.iterrows():
            model = str(row["model"])
            ax.scatter(
                row["delta_psnr_y_vs_lte"],
                row["delta_consistency_psnr_y_vs_lte"],
                s=68 if model == "SC-INR" else 42,
                color=STYLE.get(model, "#888888"),
                edgecolor="black" if model == "SC-INR" else "none",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(model, (row["delta_psnr_y_vs_lte"], row["delta_consistency_psnr_y_vs_lte"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(DISPLAY_DATASET[dataset])
        ax.set_xlabel("OOD PSNR-Y gain vs. LTE (dB)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("SC-PSNR gain vs. LTE (dB)")
    savefig(fig, out_dir / "fig_quality_consistency_tradeoff")


def write_key_findings(out_dir: Path, quality_split: pd.DataFrame, consistency_agg: pd.DataFrame) -> None:
    lines = [
        "# Seed-1 Auxiliary Metric Findings",
        "",
        "Inputs: `quality_summary.csv` and `consistency_summary.csv` from `seed1_aux_metrics_all8`.",
        "Quality OOD means average x8, x16, and x30. Consistency means average x8/x16/x30 -> x4.",
        "",
        "## SC-INR vs. LTE",
        "",
    ]
    for dataset in ["bsd100", "urban100"]:
        q_all = quality_split[
            (quality_split["model"].astype(str) == "SC-INR")
            & (quality_split["dataset"] == dataset)
        ]
        q_ood = q_all[q_all["split"] == "x8-x30"].iloc[0]
        q_id = q_all[q_all["split"] == "x4"].iloc[0]
        c = consistency_agg[
            (consistency_agg["model"].astype(str) == "SC-INR")
            & (consistency_agg["dataset"] == dataset)
        ].iloc[0]
        lines.extend([
            f"- {DISPLAY_DATASET[dataset]} x4 PSNR delta: {q_id['delta_psnr_y_vs_lte']:+.3f} dB.",
            f"- {DISPLAY_DATASET[dataset]} OOD PSNR delta: {q_ood['delta_psnr_y_vs_lte']:+.3f} dB; OOD SSIM delta: {q_ood['delta_ssim_y_vs_lte']:+.5f}.",
            f"- {DISPLAY_DATASET[dataset]} OOD texture RMSE delta: {q_ood['delta_texture_rmse_vs_lte']:+.5f}.",
            f"- {DISPLAY_DATASET[dataset]} consistency PSNR delta: {c['delta_consistency_psnr_y_vs_lte']:+.2f} dB; texture consistency RMSE delta: {c['delta_texture_consistency_rmse_vs_lte']:+.5f}.",
        ])
    lines.extend([
        "",
        "## Interpretation Caveat",
        "",
        "LTE-NoCell and LTE-FeaturePhase obtain much higher cross-scale consistency because their output is weakly conditioned on cell size. They should be treated as diagnostic controls rather than better ASISR models unless reconstruction quality and texture errors are considered jointly.",
        "",
        "## Paper Use",
        "",
        "- Use `quality_psnr_table.tex` for the compact PSNR table.",
        "- Use `quality_ssim_texture_table.tex` for perceptual/texture support.",
        "- Use `consistency_table.tex` for the same-LR cross-scale observation consistency metric.",
        "- Use `fig_quality_consistency_tradeoff.pdf` to show that SC-INR improves consistency while preserving quality better than purely removing cell response.",
        "",
    ])
    (out_dir / "key_findings.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    in_dir = args.input if args.input.is_absolute() else ROOT / args.input
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_dir(out_dir)

    _, quality_per_scale, quality_split = load_quality(in_dir)
    consistency_per_scale, consistency_agg = load_consistency(in_dir)

    quality_per_scale.to_csv(out_dir / "quality_per_scale_with_lte_deltas.csv", index=False)
    quality_split.to_csv(out_dir / "quality_id_ood_with_lte_deltas.csv", index=False)
    consistency_per_scale.to_csv(out_dir / "consistency_per_pair_with_lte_deltas.csv", index=False)
    consistency_agg.to_csv(out_dir / "consistency_avg_with_lte_deltas.csv", index=False)

    write_quality_tables(out_dir, quality_split)
    write_consistency_table(out_dir, consistency_agg)
    plot_scale_gain(out_dir, quality_per_scale)
    plot_ood_quality(out_dir, quality_split)
    plot_consistency_tradeoff(out_dir, quality_split, consistency_agg)
    write_key_findings(out_dir, quality_split, consistency_agg)

    print(f"Paper artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
