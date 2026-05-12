"""Shared paper-facing model aliases for analysis scripts.

This helper reads the project model registry and exposes checkpoint paths for
seed1 diagnostic scripts. It intentionally treats CLI names as paper-facing
display names: `SC-INR` means the final candidate, while `SC-INR-Adaptive`
means the legacy no-phase variant. Raw-result key disambiguation must be
source-aware because older files used raw key `SC-INR` for the no-phase variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs" / "registry" / "models.yaml"


STYLE = {
    "Bicubic": "#8c8c8c",
    "LIIF": "#4C72B0",
    "LIIF-EQ": "#64B5CD",
    "LTE": "#DD8452",
    "LTE-EQ": "#DDAA33",
    "LTE-NoCellPhase": "#55A868",
    "LTE-NoCell": "#55A868",
    "LTE-NoC": "#55A868",
    "LTE-PhaseZ": "#8172B2",
    "LTE-FeaturePhase": "#8172B2",
    "SC-INR-FixedOmega": "#C44E52",
    "SC-INR-Fixed": "#C44E52",
    "SC-INR-NoPhi": "#8B0000",
    "SC-INR-Adaptive": "#8B0000",
    "SC-INR-NoPhi-Signed": "#AA3377",
    "SC-INR-Signed": "#AA3377",
    "SC-INR-Adaptive-Signed": "#AA3377",
    "SC-INR": "#B22222",
    "SC-INR+PhiZ": "#B22222",
    "SC-INR-EQ": "#CC6677",
    "SC-INR-NoSinc": "#666666",
}


def load_registry() -> dict[str, dict[str, Any]]:
    with REGISTRY_PATH.open("r") as f:
        data = yaml.safe_load(f)
    return data["models"]


def analysis_aliases() -> dict[str, str]:
    """Return aliases safe for checkpoint-loading CLIs.

    Do not map raw key `SC-INR` to `SC-INR-NoPhi` here; analysis CLI users expect
    `SC-INR` to mean the current final candidate. Legacy no-phase access remains
    available through `SC-INR-NoPhi` and `SC-INR-Adaptive`.
    """

    aliases: dict[str, str] = {}
    registry = load_registry()
    for canonical, info in registry.items():
        names = {
            canonical,
            info.get("display_name"),
            info.get("paper_name"),
            info.get("short_name"),
            info.get("legacy_display_name"),
        }
        for raw_key in info.get("raw_keys", []) or []:
            if raw_key == "SC-INR" and canonical != "SC-INR":
                continue
            names.add(raw_key)
        for name in names:
            if name:
                aliases[str(name)] = canonical
    return aliases


def seed1_model_paths() -> dict[str, Path]:
    registry = load_registry()
    aliases = analysis_aliases()
    paths: dict[str, Path] = {}
    for alias, canonical in aliases.items():
        checkpoint_dir = registry.get(canonical, {}).get("checkpoint_dir_seed1")
        if checkpoint_dir:
            paths[alias] = ROOT / checkpoint_dir / "epoch-best.pth"
    return paths


MODEL_ALIASES = analysis_aliases()
MODEL_PATHS = seed1_model_paths()

LEGACY_SCINR_NOPHI_SOURCES = {
    "artifacts/raw_results/seed1/benchmark.json",
    "artifacts/raw_results/seed2/benchmark.json",
    "artifacts/raw_results/seed3/benchmark.json",
    "results/benchmark.json",
    "results/seeds/benchmark_seed2.json",
    "results/seeds/benchmark_seed3.json",
}


def is_legacy_sc_inr_nophi_source(source_path: Path | None, raw_keys: set[str]) -> bool:
    if "SC-INR+PhiZ" in raw_keys:
        return True
    if source_path is None:
        return False
    candidates = [source_path.as_posix(), source_path.resolve().as_posix()]
    return any(path.endswith(legacy) for path in candidates for legacy in LEGACY_SCINR_NOPHI_SOURCES)


def canonicalize_result_key(raw_key: str, source_path: Path | None, raw_keys: set[str]) -> str:
    """Canonicalize a raw result key with source context.

    The raw key `SC-INR` is ambiguous. It means `SC-INR-NoPhi` in legacy/core
    result files, but final `SC-INR` in cleaned final-candidate files. This
    function must be used only when the source file context is known.
    """

    if raw_key == "SC-INR" and is_legacy_sc_inr_nophi_source(source_path, raw_keys):
        return "SC-INR-NoPhi"
    return MODEL_ALIASES.get(raw_key, raw_key)
