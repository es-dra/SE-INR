#!/usr/bin/env python3
"""Build canonical benchmark tables from raw result JSON files.

The project has historical raw keys whose meaning changed over time. This
script is the paper-facing aggregation entry point: it keeps raw JSON files
unchanged, but writes explicit raw-key provenance into derived CSV tables so
later readers do not have to infer semantics from filenames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ["Set5", "Set14", "BSD100", "Urban100"]
ID_SCALES = ["x2", "x3", "x4"]
OOD_SCALES = ["x6", "x8", "x12", "x16", "x24", "x30"]
ALL_SCALES = ID_SCALES + OOD_SCALES


RAW_SOURCES = [
    {
        "seed": 1,
        "path": "artifacts/raw_results/seed1/benchmark_signed_phiz.json",
        "result_set": "seed1_signed_phiz",
        "status": "canonical_seed1_context",
        "mapping": {
            "LIIF": "LIIF",
            "LIIF-EQ": "LIIF-EQ",
            "LTE": "LTE",
            "LTE-EQ": "LTE-EQ",
            "LTE-NoCell": "LTE-NoCellPhase",
            "LTE-FeaturePhase": "LTE-PhaseZ",
            "SC-INR-Fixed": "SC-INR-FixedOmega",
            "SC-INR": "SC-INR-NoPhi",
            "SC-INR-Signed": "SC-INR-NoPhi-Signed",
            "SC-INR+PhiZ": "SC-INR",
        },
        "note": "Seed1 raw key SC-INR is legacy no-phase; SC-INR+PhiZ is final SC-INR.",
    },
    {
        "seed": 1,
        "path": "artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json",
        "result_set": "seed1_nosinc",
        "status": "canonical_seed1_ablation",
        "mapping": {"SC-INR-NoSinc": "SC-INR-NoSinc"},
        "note": "Seed1 final-candidate ablation without analytic sinc response.",
    },
    {
        "seed": 1,
        "path": "artifacts/raw_results/seed1/benchmark_sc_inr_eq.json",
        "result_set": "seed1_sc_inr_eq",
        "status": "exploratory_seed1_extension",
        "mapping": {"SC-INR-EQ": "SC-INR-EQ"},
        "note": "Exploratory Rot-E integration; not a main-method result.",
    },
    {
        "seed": 2,
        "path": "artifacts/raw_results/seed2/benchmark.json",
        "result_set": "seed2_core",
        "status": "canonical_core",
        "mapping": {"LIIF": "LIIF", "LTE": "LTE", "SC-INR-Adaptive": "SC-INR-NoPhi"},
        "note": "Core seed2 benchmark. No final SC-INR in this file.",
    },
    {
        "seed": 2,
        "path": "artifacts/raw_results/seed2/benchmark_sc_inr.json",
        "result_set": "seed2_final_sc_inr",
        "status": "canonical_final_sc_inr",
        "mapping": {"SC-INR": "SC-INR"},
        "note": "Clean raw key SC-INR means final candidate for seed2.",
    },
    {
        "seed": 3,
        "path": "artifacts/raw_results/seed3/benchmark.json",
        "result_set": "seed3_core",
        "status": "canonical_core",
        "mapping": {"LIIF": "LIIF", "LTE": "LTE", "SC-INR": "SC-INR-NoPhi"},
        "note": "Legacy seed3 core benchmark. Raw key SC-INR is no-phase here.",
    },
    {
        "seed": 3,
        "path": "artifacts/raw_results/seed3/benchmark_sc_inr.json",
        "result_set": "seed3_final_sc_inr",
        "status": "canonical_final_sc_inr",
        "mapping": {"SC-INR": "SC-INR"},
        "note": "Clean raw key SC-INR means final candidate for seed3.",
    },
]


PAPER_MAIN_MODELS = ["LIIF", "LTE", "SC-INR"]
CONTEXT_SEED1_MODELS = [
    "LIIF",
    "LIIF-EQ",
    "LTE",
    "LTE-EQ",
    "LTE-NoCellPhase",
    "LTE-PhaseZ",
    "SC-INR-FixedOmega",
    "SC-INR-NoPhi",
    "SC-INR-NoPhi-Signed",
    "SC-INR",
    "SC-INR-NoSinc",
    "SC-INR-EQ",
]


def read_registry(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "configs/registry/models.yaml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data["models"]


def split_for_scale(scale: str) -> str:
    if scale in ID_SCALES:
        return "ID"
    if scale in OOD_SCALES:
        return "OOD"
    return "OTHER"


def load_raw_rows(root: Path, registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in RAW_SOURCES:
        path = root / source["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r") as f:
            data = json.load(f)
        for raw_key, canonical_model in source["mapping"].items():
            if raw_key not in data:
                raise KeyError(f"{raw_key} missing from {path}")
            model_data = data[raw_key]
            model_info = registry.get(canonical_model, {})
            checkpoint_dir = model_info.get(f"checkpoint_dir_seed{source['seed']}", model_info.get("checkpoint_dir_seed1", ""))
            checkpoint = f"{checkpoint_dir}/epoch-best.pth" if checkpoint_dir else ""
            for dataset in BENCHMARKS:
                if dataset not in model_data:
                    raise KeyError(f"{dataset} missing for {raw_key} in {path}")
                for scale in ALL_SCALES:
                    value = model_data[dataset].get(scale)
                    psnr = float(value["psnr"] if isinstance(value, dict) else value)
                    if not math.isfinite(psnr):
                        raise ValueError(f"Non-finite PSNR for {raw_key} {dataset} {scale} in {path}")
                    rows.append(
                        {
                            "seed": source["seed"],
                            "result_set": source["result_set"],
                            "source_file": source["path"],
                            "raw_key": raw_key,
                            "raw_key_semantics": source["note"],
                            "canonical_model": canonical_model,
                            "model_status": model_info.get("status", ""),
                            "model_role": model_info.get("ablation_axis", model_info.get("note", "")),
                            "checkpoint": checkpoint,
                            "protocol_id": "benchmark_hr_downsample_4datasets_9scales_eval_full",
                            "dataset": dataset,
                            "scale": scale,
                            "split": split_for_scale(scale),
                            "psnr": f"{psnr:.6f}",
                            "finite": "true",
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict[str, Any]], split: str) -> float:
    vals = [float(r["psnr"]) for r in rows if split == "ALL" or r["split"] == split]
    if not vals:
        return float("nan")
    return mean(vals)


def summarize_seed_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    meta: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), row["canonical_model"])
        grouped[key].append(row)
        meta[key] = row

    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        seed, model = key
        sub = grouped[key]
        first = meta[key]
        out.append(
            {
                "seed": seed,
                "canonical_model": model,
                "model_status": first["model_status"],
                "result_set": first["result_set"],
                "source_file": first["source_file"],
                "raw_key": first["raw_key"],
                "id_psnr": f"{average(sub, 'ID'):.6f}",
                "ood_psnr": f"{average(sub, 'OOD'):.6f}",
                "all_psnr": f"{average(sub, 'ALL'):.6f}",
                "n_items": len(sub),
            }
        )
    return out


def rows_for_model_seed(rows: list[dict[str, Any]], model: str, seed: int) -> list[dict[str, Any]]:
    return [r for r in rows if r["canonical_model"] == model and int(r["seed"]) == seed]


def split_average_for_model_seed(rows: list[dict[str, Any]], model: str, seed: int, split: str) -> float:
    return average(rows_for_model_seed(rows, model, seed), split)


def write_paper_main(rows: list[dict[str, Any]], out_dir: Path) -> None:
    seed_summary = summarize_seed_model(rows)
    main_rows = [r for r in seed_summary if r["canonical_model"] in PAPER_MAIN_MODELS]
    write_csv(
        out_dir / "paper_main_3seed.csv",
        main_rows,
        [
            "seed",
            "canonical_model",
            "model_status",
            "result_set",
            "source_file",
            "raw_key",
            "id_psnr",
            "ood_psnr",
            "all_psnr",
            "n_items",
        ],
    )

    summary_rows: list[dict[str, Any]] = []
    for model in PAPER_MAIN_MODELS:
        model_rows = [r for r in main_rows if r["canonical_model"] == model]
        if not model_rows:
            continue
        for split_name, field in [("ID", "id_psnr"), ("OOD", "ood_psnr"), ("ALL", "all_psnr")]:
            vals = [float(r[field]) for r in model_rows]
            summary_rows.append(
                {
                    "canonical_model": model,
                    "split": split_name,
                    "mean_psnr": f"{mean(vals):.6f}",
                    "std_psnr": f"{stdev(vals):.6f}" if len(vals) > 1 else "nan",
                    "n_seeds": len(vals),
                    "min_psnr": f"{min(vals):.6f}",
                    "max_psnr": f"{max(vals):.6f}",
                }
            )
    write_csv(
        out_dir / "paper_main_3seed_summary.csv",
        summary_rows,
        ["canonical_model", "split", "mean_psnr", "std_psnr", "n_seeds", "min_psnr", "max_psnr"],
    )

    paired_rows: list[dict[str, Any]] = []
    for split_name, field in [("ID", "id_psnr"), ("OOD", "ood_psnr"), ("ALL", "all_psnr")]:
        sc = {int(r["seed"]): split_average_for_model_seed(rows, "SC-INR", int(r["seed"]), split_name) for r in main_rows if r["canonical_model"] == "SC-INR"}
        for baseline in ["LIIF", "LTE"]:
            base = {int(r["seed"]): split_average_for_model_seed(rows, baseline, int(r["seed"]), split_name) for r in main_rows if r["canonical_model"] == baseline}
            shared = sorted(set(sc) & set(base))
            deltas = [sc[s] - base[s] for s in shared]
            paired_rows.append(
                {
                    "model": "SC-INR",
                    "baseline": baseline,
                    "split": split_name,
                    "mean_delta": f"{mean(deltas):.6f}",
                    "std_delta": f"{stdev(deltas):.6f}" if len(deltas) > 1 else "nan",
                    "n_seed_pairs": len(deltas),
                    "seeds": ";".join(map(str, shared)),
                }
            )
    write_csv(
        out_dir / "paper_main_3seed_paired_delta.csv",
        paired_rows,
        ["model", "baseline", "split", "mean_delta", "std_delta", "n_seed_pairs", "seeds"],
    )


def write_context_seed1(rows: list[dict[str, Any]], out_dir: Path) -> None:
    seed1 = [r for r in rows if int(r["seed"]) == 1 and r["canonical_model"] in CONTEXT_SEED1_MODELS]
    seed_summary = summarize_seed_model(seed1)
    order = {m: i for i, m in enumerate(CONTEXT_SEED1_MODELS)}
    seed_summary.sort(key=lambda r: order.get(r["canonical_model"], 999))
    lte_by_split = {
        "ID": next(float(r["id_psnr"]) for r in seed_summary if r["canonical_model"] == "LTE"),
        "OOD": next(float(r["ood_psnr"]) for r in seed_summary if r["canonical_model"] == "LTE"),
        "ALL": next(float(r["all_psnr"]) for r in seed_summary if r["canonical_model"] == "LTE"),
    }
    out: list[dict[str, Any]] = []
    for row in seed_summary:
        out.append(
            {
                **row,
                "delta_id_vs_lte": f"{float(row['id_psnr']) - lte_by_split['ID']:.6f}",
                "delta_ood_vs_lte": f"{float(row['ood_psnr']) - lte_by_split['OOD']:.6f}",
                "delta_all_vs_lte": f"{float(row['all_psnr']) - lte_by_split['ALL']:.6f}",
            }
        )
    write_csv(
        out_dir / "paper_context_seed1.csv",
        out,
        [
            "seed",
            "canonical_model",
            "model_status",
            "result_set",
            "source_file",
            "raw_key",
            "id_psnr",
            "ood_psnr",
            "all_psnr",
            "n_items",
            "delta_id_vs_lte",
            "delta_ood_vs_lte",
            "delta_all_vs_lte",
        ],
    )


def write_readme(out_dir: Path) -> None:
    text = """# Canonical Benchmarks

本目录是当前论文 benchmark 的活跃入口。它不改动 raw JSON，而是把每一行
paper-facing 结果显式展开为 `source_file`、`raw_key`、`canonical_model`、
`model_status`、`seed`、`dataset`、`scale`、`checkpoint`、`protocol_id` 和
`raw_key_semantics`，避免后续再从文件名或 raw key 猜模型语义。

## 文件

- `benchmark_all_models_long.csv`：所有 benchmark 记录的 canonical long table。
- `benchmark_by_seed_model.csv`：每个 seed/model 的 ID/OOD/ALL 汇总。
- `paper_main_3seed.csv`：论文主表使用的 LIIF/LTE/SC-INR per-seed benchmark。
- `paper_main_3seed_summary.csv`：论文主表 mean/std 汇总。
- `paper_main_3seed_paired_delta.csv`：`SC-INR` 相对 LIIF/LTE 的 paired delta。
- `paper_context_seed1.csv`：seed1 context/diagnostic 模型，包括 NoPhi、NoSinc、
  EQ 和 LTE-side diagnostics。

## 命名 caveat

`artifacts/raw_results/seed1/benchmark_signed_phiz.json` 中的 raw key
`SC-INR` 表示 `SC-INR-NoPhi`，seed1 final candidate 是 raw key
`SC-INR+PhiZ`。`seed2/benchmark_sc_inr.json` 和
`seed3/benchmark_sc_inr.json` 中的 clean raw key `SC-INR` 才表示 final
candidate。论文表格、claim 和后续 agent 恢复应优先读取本目录，而不是直接读取
raw JSON key。
"""
    (out_dir / "README_zh.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical benchmark CSV tables.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/derived/benchmarks")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out if args.out.is_absolute() else root / args.out
    registry = read_registry(root)
    rows = load_raw_rows(root, registry)
    write_csv(
        out_dir / "benchmark_all_models_long.csv",
        rows,
        [
            "seed",
            "result_set",
            "source_file",
            "raw_key",
            "raw_key_semantics",
            "canonical_model",
            "model_status",
            "model_role",
            "checkpoint",
            "protocol_id",
            "dataset",
            "scale",
            "split",
            "psnr",
            "finite",
        ],
    )
    write_csv(
        out_dir / "benchmark_by_seed_model.csv",
        summarize_seed_model(rows),
        [
            "seed",
            "canonical_model",
            "model_status",
            "result_set",
            "source_file",
            "raw_key",
            "id_psnr",
            "ood_psnr",
            "all_psnr",
            "n_items",
        ],
    )
    write_paper_main(rows, out_dir)
    write_context_seed1(rows, out_dir)
    write_readme(out_dir)
    print(f"Canonical benchmark tables written to {out_dir}")


if __name__ == "__main__":
    main()
