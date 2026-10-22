"""Extract CPU-only spectral statistics from frozen Set14 float16 predictions.

No model packages are imported and no inference is performed. Original prediction
and HR files are read as bytes, so each recorded SHA-256 identifies the exact
bytes decoded. This is a new analysis of saved, quantized outputs, not a rerun of
the original float32 inference metrics.
"""

import argparse
import hashlib
import io
import itertools
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


H, SH = 240, 4
N = H - 2 * SH
NORMALIZER = float(N**4 * 3)
STAT_NAMES = ["C", "U", "E", "V", "K", "D", "Z", "G"]
MODELS = ("liif", "lte", "sc-inr")
SEEDS = (1, 2)
SCALES = (2, 3, 4, 6)
MU_KEYS = ("0.5", "1.3333", "2.0")
REQUIRED_KEYS = ("1.0",) + MU_KEYS
FILE_PATTERN = re.compile(r"^Set14_(.+)_x(2|3|4|6)_(liif|lte|sc-inr)_s([12])\.npz$")
REFERENCE_SHA256 = {
    "01-cell-mismatch/run.py": "91ae35edb38fd28ac2e0b12b35c3239c4bd19ddd6dcf3e8fbce90af0fac5ab83",
    "01-cell-mismatch/transfer.py": "b0491871532015458584fc722fca716dbaf60b4b6406fc9a8896ef0bbd5214ed",
    "01-cell-mismatch/protocol.json": "507982131d56e0276354cbf66d8b3b73490cbc70089faa4099de6a6ce14c52a1",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")


def geometry():
    frequency = np.fft.fftfreq(N)
    radius = np.maximum(np.abs(frequency[:, None]), np.abs(frequency[None, :]))
    bins = np.floor(radius * 32).astype(np.int64)
    if bins.min() != 0 or bins.max() != 16:
        raise ValueError("Unexpected FFT bin coverage")
    counts = np.bincount(bins.ravel(), minlength=17)
    counts[0] -= 1  # DC is stored separately, never counted twice.
    if int(counts.sum()) + 1 != N**2 or np.any(counts <= 0):
        raise ValueError("FFT bins do not partition all non-DC coefficients")
    return bins, counts


def spectrum(array):
    return np.fft.fft2(array.astype(np.float64)[SH:H-SH, SH:H-SH], axes=(0, 1))


def validate_prediction(array, label):
    if array.shape != (H, H, 3):
        raise ValueError(f"{label}: expected shape {(H, H, 3)}, got {array.shape}")
    if array.dtype != np.float16:
        raise ValueError(f"{label}: expected frozen float16 prediction, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite prediction")


def summarize_pair(target, native, perturbed, bins):
    """Return (17 x 8 bin statistics, 8 DC statistics), with e=T-P."""
    error = target - native
    fields = (
        (np.conj(error) * native).real,
        np.abs(native)**2,
        np.abs(target)**2,
        np.abs(error)**2,
        (np.conj(native) * perturbed).real,
        np.abs(target - perturbed)**2 - np.abs(error)**2,
        np.abs(perturbed)**2,
        (np.conj(target) * perturbed).real,
    )
    values = np.empty((17, len(STAT_NAMES)), dtype=np.float64)
    dc = np.empty(len(STAT_NAMES), dtype=np.float64)
    for column, field in enumerate(fields):
        rgb_sum = field.sum(axis=2)
        dc[column] = rgb_sum[0, 0] / NORMALIZER
        values[:, column] = np.bincount(
            bins.ravel(), weights=rgb_sum.ravel(), minlength=17
        ) / NORMALIZER
        values[0, column] -= dc[column]
    return values, dc


def spatial_pair(target, native, perturbed):
    """The same eight inner products in spatial coordinates (RGB mean MSE)."""
    target, native, perturbed = (
        array.astype(np.float64)[SH:H-SH, SH:H-SH]
        for array in (target, native, perturbed)
    )
    error = target - native
    return np.asarray([
        np.mean(error * native), np.mean(native**2), np.mean(target**2),
        np.mean(error**2), np.mean(native * perturbed),
        np.mean((target - perturbed)**2 - error**2),
        np.mean(perturbed**2), np.mean(target * perturbed),
    ], dtype=np.float64)


class Checks:
    def __init__(self):
        self.atol = 2e-12
        self.rtol = 2e-10
        self.metrics = {}
        self.finite = True
        self.failures = []

    def compare(self, name, actual, expected, label):
        actual, expected = np.asarray(actual), np.asarray(expected)
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        self.finite = self.finite and finite
        if not finite:
            self.failures.append(f"{label}: {name}: non-finite values")
            return
        difference = np.abs(actual - expected)
        tolerance = self.atol + self.rtol * np.maximum(np.abs(actual), np.abs(expected))
        maximum = float(np.max(difference))
        ratio = float(np.max(difference / tolerance))
        metric = self.metrics.setdefault(name, {
            "comparisons": 0, "max_abs_error": 0.0, "max_tolerance_ratio": 0.0,
            "worst_abs_error_record": None, "passed": True,
        })
        metric["comparisons"] += int(difference.size)
        if maximum > metric["max_abs_error"]:
            metric["max_abs_error"] = maximum
            metric["worst_abs_error_record"] = label
        metric["max_tolerance_ratio"] = max(metric["max_tolerance_ratio"], ratio)
        metric["passed"] = metric["passed"] and ratio <= 1.0
        if ratio > 1.0 and len(self.failures) < 20:
            self.failures.append(f"{label}: {name}: tolerance ratio {ratio:.6g}")


def inventory(pred_dir, hr_dir):
    paths = sorted(pred_dir.glob("*.npz"))
    entries = []
    for path in paths:
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected prediction filename: {path.name}")
        image, scale, model, seed = match.groups()
        entries.append((path, image, model, int(seed), int(scale)))
    images = sorted({entry[1] for entry in entries})
    if len(images) != 14 or len(entries) != 336:
        raise ValueError(f"Expected 14 images and 336 prediction files; got {len(images)} and {len(entries)}")
    expected = set(itertools.product(images, MODELS, SEEDS, SCALES))
    actual = {entry[1:] for entry in entries}
    if len(actual) != len(entries) or actual != expected:
        raise ValueError("Prediction inventory does not match the complete 14x3x2x4 design")
    for image in images:
        if not (hr_dir / f"{image}.png").is_file():
            raise ValueError(f"Missing HR PNG: {image}.png")
    return entries, images


def extract(args, output, started):
    pred_dir, hr_dir = Path(args.pred_dir).resolve(), Path(args.hr_dir).resolve()
    if not pred_dir.is_dir() or not hr_dir.is_dir():
        raise ValueError("--pred-dir and --hr-dir must be existing directories")
    entries, images = inventory(pred_dir, hr_dir)
    bins, counts = geometry()
    checks = Checks()
    metadata = {
        "H": H, "shave": SH, "N": N, "normalizer": NORMALIZER,
        "normalization": "sum over FFT coefficients and RGB / (N**4 * 3), once",
        "fft": "numpy.fft.fft2, float64 input, unnormalized forward transform",
        "r_definition": "max(abs(fx),abs(fy)), cycles per output pixel",
        "bin_definition": "floor(r*32); bins 0..16; DC excluded and stored separately",
        "bin_coefficient_counts": counts.tolist(), "dc_coefficient_count": 1,
        "last_bin": "16 contains the r=0.5 Nyquist outer row/column",
        "error_definition": "e=T-P, P=native output, Pm=perturbed output",
        "stat_definitions": {
            "C": "Re<e,P>", "U": "|P|^2", "E": "|T|^2", "V": "|e|^2",
            "K": "Re<P,Pm>", "D": "|T-Pm|^2-|e|^2", "Z": "|Pm|^2",
            "G": "Re<T,Pm>",
        },
        "target_preprocessing": "center-crop 240 RGB; float32 /255 as in original run; cast float64 for FFT",
        "prediction_precision": "frozen float16, cast to float64; no clipping",
        "native_mu": 1.0, "analyzed_mu": [float(key) for key in MU_KEYS],
        "scope": "no_model_calls", "training_runs": 0, "inference_runs": 0,
    }
    manifest = {
        "schema_version": 1, "scope": "no_model_calls", "training_runs": 0,
        "inference_runs": 0, "pred_dir": str(pred_dir), "hr_dir": str(hr_dir),
        "extractor": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).read_bytes())},
        "interpretation_reference_sha256": REFERENCE_SHA256,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": Image.__version__},
        "hr": [], "predictions": [],
    }
    targets = {}
    for image in images:
        path = hr_dir / f"{image}.png"
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as original:
            rgb = original.convert("RGB")
            width, height = rgb.size
            if min(width, height) < H:
                raise ValueError(f"HR image too small for original center crop: {path}")
            left, top = (width-H)//2, (height-H)//2
            array = np.asarray(rgb.crop((left, top, left+H, top+H)), dtype=np.float32) / np.float32(255)
        targets[image] = (array, spectrum(array))
        manifest["hr"].append({"image": image, "path": str(path), "sha256": sha256(raw), "bytes": len(raw), "size": [width, height]})
    records = []
    for index, (path, image, model, seed, scale) in enumerate(entries, start=1):
        raw = path.read_bytes()
        target_array, target = targets[image]
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            keys = sorted(saved.files)
            missing = sorted(set(REQUIRED_KEYS) - set(keys))
            if missing:
                raise ValueError(f"{path.name}: missing cells {missing}")
            native_array = saved["1.0"]
            validate_prediction(native_array, f"{path.name}:1.0")
            native = spectrum(native_array)
            for key in MU_KEYS:
                perturbed_array = saved[key]
                label = f"{path.name}:{key}"
                validate_prediction(perturbed_array, label)
                values, dc = summarize_pair(target, native, spectrum(perturbed_array), bins)
                all_values = np.vstack((values, dc))
                checks.compare("B_equals_E_minus_V", all_values[:, 1] + 2*all_values[:, 0], all_values[:, 2] - all_values[:, 3], label)
                spectral_total = values.sum(axis=0) + dc
                spatial_total = spatial_pair(target_array, native_array, perturbed_array)
                for column, name in enumerate(STAT_NAMES):
                    checks.compare(f"Parseval_{name}", spectral_total[column], spatial_total[column], label)
                records.append({
                    "dataset": "Set14", "image": image, "model": model, "seed": seed,
                    "scale": scale, "mu": float(key), "bins": values.tolist(), "dc": dc.tolist(),
                })
        manifest["predictions"].append({"path": str(path), "sha256": sha256(raw), "bytes": len(raw), "keys": keys, "analyzed_keys": list(REQUIRED_KEYS)})
        if index % 24 == 0 or index == len(entries):
            print(f"Extracted {index}/{len(entries)} prediction files ({time.perf_counter()-started:.1f}s)", flush=True)
    expected_records = 336 * len(MU_KEYS)
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} records; got {len(records)}")
    elapsed = time.perf_counter() - started
    manifest["counts"] = {"images": len(images), "prediction_files": len(entries), "models": len(MODELS), "seeds": len(SEEDS), "scales": len(SCALES), "non_native_cells": len(MU_KEYS), "records": len(records)}
    manifest["elapsed_seconds"] = elapsed
    check = {
        "schema_version": 1, "passed": checks.finite and not checks.failures,
        "scope": "no_model_calls", "finite": checks.finite,
        "tolerances": {"absolute": checks.atol, "relative": checks.rtol, "rule": "abs(a-b) <= atol + rtol*max(abs(a),abs(b))"},
        "checks": checks.metrics, "coverage": manifest["counts"],
        "coverage_passed": True, "spatial_coefficients": N**2,
        "non_dc_coefficients": int(counts.sum()), "dc_coefficients": 1,
        "nyquist_outer_bin_coefficients": int(counts[-1]),
        "failures": checks.failures, "elapsed_seconds": elapsed,
    }
    write_json(output / "stats.json", {"schema_version": 1, "stat_names": STAT_NAMES, "metadata": metadata, "records": records})
    write_json(output / "source_manifest.json", manifest)
    write_json(output / "check.json", check)
    if not check["passed"]:
        raise RuntimeError("Numerical checks failed; inspect check.json")
    print(f"PASS: {len(records)} records; {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--out", required=True, help="New output directory; its parent must exist")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    for source in (Path(args.pred_dir).resolve(), Path(args.hr_dir).resolve()):
        if output == source or source in output.parents:
            parser.error("--out must be outside both read-only input directories")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        extract(args, output, started)
    except Exception as error:
        failure_path = output / "check.json"
        if not failure_path.exists():
            write_json(failure_path, {"schema_version": 1, "passed": False, "scope": "no_model_calls", "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.perf_counter()-started})
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
