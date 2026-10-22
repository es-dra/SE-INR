"""Retrospective CPU-only real projections of frozen Set14 predictions.

No torch, model loading, training, or inference. A/B/C/D form a diamond:
A (r-bin/RGB) is refined by B (r-bin/channel) and C (frequency/RGB),
and D (frequency/channel) refines both. B and C are not nested.
All fits use the native and perturbed outputs from the same image/seed;
lower residuals from more flexible fits are descriptive, not predictive evidence.
"""

import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

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
MODELS = ("liif", "lte", "sc-inr")
SEEDS, SCALES = (1, 2), (2, 3, 4, 6)
MU_KEYS = ("0.5", "1.3333", "2.0")
REQUIRED_KEYS = ("1.0",) + MU_KEYS
BASE_NAMES = ["C", "U", "E", "V", "K", "D", "Z", "G"]
LAYER_NAMES = [
    "epsilon_energy", "e_epsilon", "P_epsilon", "scalar_change_energy",
    "scalar_error_cross", "scalar_cost", "epsilon_error_cross",
    "epsilon_scalar_cross", "H", "D", "change_energy", "error_change_cross",
    "negative_w_native_energy", "negative_w_perturbed_energy",
    "negative_w_coefficient_count", "zero_P_new_output_energy",
    "zero_P_coefficient_count", "small_P_native_energy", "small_P_output_energy",
    "small_P_coefficient_count",
]
FILE_PATTERN = re.compile(r"^Set14_(.+)_x(2|3|4|6)_(liif|lte|sc-inr)_s([12])\.npz$")
# Flag only. No coefficient, denominator, or fitted gain is truncated at this threshold.
SMALL_POWER = 1e-24


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")


class Checks:
    def __init__(self):
        self.atol, self.rtol = 2e-12, 2e-10
        self.metrics, self.failures = {}, []

    def compare(self, name, actual, expected, label):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
            self.failures.append(f"{label}: {name}: non-finite")
            return
        difference = np.abs(actual - expected)
        tolerance = self.atol + self.rtol * np.maximum(np.abs(actual), np.abs(expected))
        maximum, ratio = float(np.max(difference)), float(np.max(difference / tolerance))
        metric = self.metrics.setdefault(name, {
            "comparisons": 0, "max_abs_error": 0., "max_tolerance_ratio": 0.,
            "worst_record": None, "passed": True,
        })
        metric["comparisons"] += int(difference.size)
        if maximum > metric["max_abs_error"]:
            metric["max_abs_error"], metric["worst_record"] = maximum, label
        metric["max_tolerance_ratio"] = max(metric["max_tolerance_ratio"], ratio)
        metric["passed"] = metric["passed"] and ratio <= 1.
        if ratio > 1. and len(self.failures) < 30:
            self.failures.append(f"{label}: {name}: tolerance ratio {ratio:.6g}")

    def nonnegative(self, name, array, label):
        array = np.asarray(array)
        self.compare(name, np.minimum(array, 0), np.zeros_like(array), label)


def geometry():
    frequency = np.fft.fftfreq(N)
    radius = np.maximum(np.abs(frequency[:, None]), np.abs(frequency[None, :]))
    bands = np.floor(radius * 32).astype(np.int64)
    bands[0, 0] = 17  # Separate DC; output bins 0..16 remain strictly non-DC.
    counts = np.bincount(bands.ravel(), minlength=18)
    if counts[-1] != 1 or counts.sum() != N**2 or np.any(counts <= 0):
        raise ValueError("Invalid partition of the Fourier plane")
    spatial_groups = np.arange(N*N).reshape(N, N)
    channels = np.arange(3)[None, None, :]
    groups = {
        "A": np.broadcast_to(bands[:, :, None], (N, N, 3)).ravel(),
        "B": (3 * bands[:, :, None] + channels).ravel(),
        "C": np.broadcast_to(spatial_groups[:, :, None], (N, N, 3)).ravel(),
        "D": (3 * spatial_groups[:, :, None] + channels).ravel(),
    }
    return bands, counts, groups


def aggregate(field, bands, normalize=True):
    """Return 17 non-DC r-bins followed by DC, with each channel counted once."""
    summed = np.asarray(field, dtype=np.float64).sum(axis=2)
    out = np.bincount(bands.ravel(), weights=summed.ravel(), minlength=18)
    return out / NORMALIZER if normalize else out


def spectrum(array):
    return np.fft.fft2(array.astype(np.float64)[SH:H-SH, SH:H-SH], axes=(0, 1))


def validate_prediction(array, label):
    if array.shape != (H, H, 3) or array.dtype != np.float16:
        raise ValueError(f"{label}: expected {(H,H,3)} float16, got {array.shape} {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite prediction")


def base_stats(target, native, perturbed, bands):
    error = target - native
    return np.stack([aggregate(field, bands) for field in (
        (np.conj(error) * native).real, np.abs(native)**2, np.abs(target)**2,
        np.abs(error)**2, (np.conj(native) * perturbed).real,
        np.abs(target-perturbed)**2 - np.abs(error)**2,
        np.abs(perturbed)**2, (np.conj(target) * perturbed).real,
    )], axis=1)


def project(native, perturbed, group):
    """Real least squares using unit group vectors; only exact zero gets w=0."""
    p, m = native.ravel(), perturbed.ravel()
    power = np.bincount(group, weights=np.abs(p)**2)
    nonzero = power > 0
    norm = np.sqrt(power)
    q = np.divide(p, norm[group], out=np.zeros_like(p), where=nonzero[group])
    coefficient = np.bincount(group, weights=(np.conj(q)*m).real, minlength=len(power))
    gain = np.divide(coefficient, norm, out=np.zeros_like(power), where=nonzero)
    fit = q * coefficient[group]
    if not (np.isfinite(fit).all() and np.isfinite(gain).all()):
        raise ValueError("Non-finite projection: no denominator or gain clipping is allowed")
    nz = power[nonzero]
    guard = {
        "groups": len(power), "zero_power_groups": int((~nonzero).sum()),
        "positive_power_groups_at_or_below_1e-24": int(((power <= SMALL_POWER) & nonzero).sum()),
        "minimum_positive_raw_group_power": float(nz.min()) if len(nz) else None,
        "maximum_abs_gain": float(np.max(np.abs(gain))),
        "negative_gain_groups": int((gain < 0).sum()),
        "clipped_groups": 0,
    }
    return fit.reshape(native.shape), gain[group].reshape(native.shape), guard


def layer_stats(target, native, perturbed, fit, gains, bands, checks, label):
    error, epsilon, scalar_change = target-native, perturbed-fit, native-fit
    change = perturbed-native
    eps_energy = np.abs(epsilon)**2
    e_eps = (np.conj(error)*epsilon).real
    p_eps = (np.conj(native)*epsilon).real
    scalar_energy = np.abs(scalar_change)**2
    scalar_cross = 2*(np.conj(error)*scalar_change).real
    residual_cross = -2*(np.conj(scalar_change)*epsilon).real
    scalar_cost = scalar_energy + scalar_cross
    actual = np.abs(target-perturbed)**2 - np.abs(error)**2
    remainder = eps_energy - 2*e_eps + residual_cross
    npower, mpower = np.abs(native)**2, np.abs(perturbed)**2
    negative, zero = gains < 0, npower == 0
    small = (npower <= SMALL_POWER) & (~zero)
    fields = [
        eps_energy, e_eps, p_eps, scalar_energy, scalar_cross, scalar_cost,
        -2*e_eps, residual_cross, remainder, actual, np.abs(change)**2,
        -2*(np.conj(error)*change).real,
        np.where(negative, npower, 0), np.where(negative, mpower, 0), negative,
        np.where(zero, mpower, 0), zero,
        np.where(small, npower, 0), np.where(small, mpower, 0), small,
    ]
    count_fields = {14, 16, 19}
    values = np.stack([aggregate(field, bands, normalize=i not in count_fields)
                       for i, field in enumerate(fields)], axis=1)
    checks.compare("D_equals_scalar_cost_plus_H", values[:,9], values[:,5]+values[:,8], label)
    checks.compare("D_equals_change_energy_plus_error_cross", values[:,9], values[:,10]+values[:,11], label)
    checks.compare("scalar_cost_direct", values[:,5], aggregate(np.abs(target-fit)**2-np.abs(error)**2,bands),label)
    checks.compare("P_epsilon_orthogonality", values[:,2], np.zeros(18), label)
    checks.compare("scalar_epsilon_orthogonality", values[:,7], np.zeros(18), label)
    checks.compare("change_energy_Pythagoras", values[:,10], values[:,3]+values[:,0], label)
    # Fourier symmetry is preserved by these real projections, including per-frequency fits.
    mirror = (-np.arange(N)) % N
    checks.compare("fit_conjugate_symmetry", fit / (N*N), np.conj(fit[mirror][:,mirror]) / (N*N), label)
    return values, epsilon


def inventory(pred_dir, hr_dir):
    entries = []
    for path in sorted(pred_dir.glob("*.npz")):
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected prediction filename: {path.name}")
        image, scale, model, seed = match.groups()
        entries.append((path,image,model,int(seed),int(scale)))
    images = sorted({entry[1] for entry in entries})
    if len(images) != 14 or len(entries) != 336:
        raise ValueError(f"Expected 14 images and 336 files, got {len(images)} and {len(entries)}")
    if {entry[1:] for entry in entries} != set(itertools.product(images,MODELS,SEEDS,SCALES)):
        raise ValueError("Incomplete or duplicate image/model/seed/scale inventory")
    for image in images:
        if not (hr_dir / f"{image}.png").is_file():
            raise ValueError(f"Missing HR PNG {image}")
    return entries, images


def keyed(record):
    return (record["image"],record["model"],record["seed"],record["scale"],record["mu"])


def extract(args, output, started):
    pred_dir, hr_dir = Path(args.pred_dir).resolve(), Path(args.hr_dir).resolve()
    entries, images = inventory(pred_dir, hr_dir)
    reference_paths = {name: Path(getattr(args,name)).resolve()
                       for name in ("reference_manifest", "reference_stats")}
    reference_raw = {name: path.read_bytes() for name,path in reference_paths.items()}
    ref_manifest = json.loads(reference_raw["reference_manifest"])
    ref_stats = json.loads(reference_raw["reference_stats"])
    if ref_stats["stat_names"] != BASE_NAMES:
        raise ValueError("Reference statistic columns differ from the expected frozen extraction")
    for name, value in (("H",H),("shave",SH),("N",N),("normalizer",NORMALIZER)):
        if ref_stats["metadata"][name] != value:
            raise ValueError(f"Reference geometry/normalization mismatch: {name}")
    reference = {keyed(row):row for row in ref_stats["records"]}
    if len(reference) != 1008 or len(ref_stats["records"]) != 1008:
        raise ValueError("Reference extraction must contain exactly 1008 unique records")
    expected_pred = {Path(row["path"]).name:row for row in ref_manifest["predictions"]}
    expected_hr = {row["image"]:row for row in ref_manifest["hr"]}
    if set(expected_pred) != {row[0].name for row in entries} or set(expected_hr) != set(images):
        raise ValueError("Input inventory differs from reference manifest")
    bands, counts, groups = geometry()
    checks, records, targets = Checks(), [], {}
    manifest = {
        "schema_version": 1, "scope": "saved_predictions_cpu_only", "training_runs": 0,
        "inference_runs": 0, "started_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).read_bytes())},
        "references": {name:{"path":str(reference_paths[name]),"sha256":sha256(raw)} for name,raw in reference_raw.items()},
        "software":{"python":sys.version.split()[0],"numpy":np.__version__,"pillow":Image.__version__},
        "predictions":[], "hr":[], "all_input_hashes_match_reference":True,
        "thread_environment":{key:os.environ[key] for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")},
    }
    for image in images:
        path = hr_dir / f"{image}.png"
        raw = path.read_bytes()
        if sha256(raw) != expected_hr[image]["sha256"]:
            raise ValueError(f"HR SHA mismatch: {path}")
        with Image.open(io.BytesIO(raw)) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            if min(width,height) < H:
                raise ValueError(f"HR image too small: {path}")
            left,top = (width-H)//2,(height-H)//2
            array = np.asarray(rgb.crop((left,top,left+H,top+H)),dtype=np.float32)/np.float32(255)
        targets[image] = (array,spectrum(array))
        manifest["hr"].append({"image":image,"path":str(path),"sha256":sha256(raw),"bytes":len(raw),"reference_match":True})
    for index,(path,image,model,seed,scale) in enumerate(entries,start=1):
        raw = path.read_bytes()
        if sha256(raw) != expected_pred[path.name]["sha256"]:
            raise ValueError(f"Prediction SHA mismatch: {path}")
        target_array,target = targets[image]
        with np.load(io.BytesIO(raw),allow_pickle=False) as saved:
            keys = sorted(saved.files)
            if not set(REQUIRED_KEYS).issubset(keys):
                raise ValueError(f"Missing cell outputs: {path}")
            native_array = saved["1.0"]
            validate_prediction(native_array,f"{path.name}:native")
            native = spectrum(native_array)
            for key in MU_KEYS:
                label = f"{path.name}:{key}"
                perturbed_array = saved[key]
                validate_prediction(perturbed_array,label)
                perturbed = spectrum(perturbed_array)
                base = base_stats(target,native,perturbed,bands)
                ref = reference[(image,model,seed,scale,float(key))]
                checks.compare("reference_raw_statistics",base,np.vstack((ref["bins"],ref["dc"])),label)
                spatial_error = target_array.astype(np.float64)[SH:H-SH,SH:H-SH]-native_array.astype(np.float64)[SH:H-SH,SH:H-SH]
                spatial_delta = perturbed_array.astype(np.float64)[SH:H-SH,SH:H-SH]-native_array.astype(np.float64)[SH:H-SH,SH:H-SH]
                checks.compare("Parseval_actual_cost",base[:,5].sum(),np.mean(spatial_delta**2-2*spatial_error*spatial_delta),label)
                layers, residuals = {}, {}
                for layer, group in groups.items():
                    fit,gains,guard = project(native,perturbed,group)
                    values,epsilon = layer_stats(target,native,perturbed,fit,gains,bands,checks,f"{label}:{layer}")
                    checks.compare("layer_D_equals_base_D",values[:,9],base[:,5],f"{label}:{layer}")
                    layers[layer] = {"bins":values[:17].tolist(),"dc":values[17].tolist(),"guard":guard}
                    residuals[layer] = epsilon
                for coarse,fine in (("A","B"),("A","C"),("B","D"),("C","D")):
                    improvement = aggregate(np.abs(residuals[coarse])**2-np.abs(residuals[fine])**2,bands)
                    extra_fit = aggregate(np.abs(residuals[coarse]-residuals[fine])**2,bands)
                    checks.compare(f"nested_{coarse}_{fine}_Pythagoras",improvement,extra_fit,label)
                    checks.nonnegative(f"nested_{coarse}_{fine}_energy",improvement,label)
                power = np.abs(native)**2
                valid = power > 0
                unit_p = np.divide(native,np.sqrt(power),out=np.zeros_like(native),where=valid)
                quadrature = (np.conj(unit_p)*perturbed).imag**2
                new_frequency = np.where(valid,0,np.abs(perturbed)**2)
                checks.compare("D_quadrature_plus_zero_native_energy",aggregate(np.abs(residuals["D"])**2,bands),aggregate(quadrature+new_frequency,bands),label)
                checks.compare("D_quadrature_real_orthogonality",aggregate((np.conj(unit_p)*residuals["D"]).real,bands),np.zeros(18),label)
                records.append({
                    "dataset":"Set14","image":image,"model":model,"seed":seed,"scale":scale,"mu":float(key),
                    "base_bins":base[:17].tolist(),"base_dc":base[17].tolist(),"layers":layers,
                    "D_quadrature_energy_bins":aggregate(quadrature,bands)[:17].tolist(),
                    "D_quadrature_energy_dc":float(aggregate(quadrature,bands)[17]),
                })
        manifest["predictions"].append({"path":str(path),"sha256":sha256(raw),"bytes":len(raw),"keys":keys,"reference_match":True})
        if index%24 == 0 or index == len(entries):
            print(f"Projected {index}/{len(entries)} files ({time.perf_counter()-started:.1f}s)",flush=True)
    manifest["counts"] = {"images":14,"prediction_files":336,"records":len(records),"layers":4,"non_dc_bins":17}
    manifest["elapsed_seconds"] = time.perf_counter()-started
    metadata = {
        "H":H,"shave":SH,"N":N,"normalizer":NORMALIZER,
        "normalization":"sum FFT coefficient/channel contributions divided once by N**4*3; count columns are unnormalized",
        "r_definition":"max(abs(fx),abs(fy)); bin=floor(32*r); bin16 is r=0.5 Nyquist edge",
        "bin_coefficient_counts":counts[:17].tolist(),"DC":"separate from every bin",
        "scope_primary":"bins0..15, 0<r<0.5","scope_audit":"bins0..16, complete non-DC",
        "fits":{"A":"one real w per image/seed/r-bin, shared RGB","B":"one real w per image/seed/r-bin/channel","C":"one real w per image/seed/Fourier point, shared RGB","D":"one real w per image/seed/Fourier point/channel"},
        "nesting_edges":[["A","B"],["A","C"],["B","D"],["C","D"]],
        "non_nesting":"B and C are not nested. Their difference is not an additive decomposition.",
        "fit_target":"Pm; real unregularized least squares fitted to native and perturbed outputs from same example",
        "definitions":{"P":"native output FFT","Pm":"perturbed output FFT","T":"target FFT","e":"T-P","epsilon":"Pm-fitted(P)","scalar_change":"P-fitted(P)","scalar_cost":"||T-fitted(P)||^2-||e||^2","H":"epsilon_energy-2 Re<e,epsilon>-2 Re<scalar_change,epsilon>","D":"||T-Pm||^2-||e||^2","D_quadrature":"for nonzero P: |Pm|^2*sin^2(arg(Pm)-arg(P)); negative real gain (pi phase) is fitted, not in this residual"},
        "small_power_guard":{"threshold_raw_squared_FFT":SMALL_POWER,"effect":"flag only; no truncation, epsilon clipping, or gain clipping","zero_case":"w=0 only when group native power exactly zero; residual retained"},
        "interpretation":"Descriptive fitted-subspace energy accounting, not causal attribution or prospective prediction. Quadrature is geometric, not a proven phase mechanism. Saved float16 outputs may have quantization-level spectral residues.",
        "prediction_precision":"frozen float16 cast float64; target float32 /255 then float64 FFT; no clipping",
        "scope":"no_model_calls","training_runs":0,"inference_runs":0,
    }
    check = {"schema_version":1,"passed":not checks.failures,"checks":checks.metrics,
             "failures":checks.failures,"coverage":manifest["counts"],
             "all_input_hashes_match_reference":True,
             "tolerances":{"atol":checks.atol,"rtol":checks.rtol},
             "elapsed_seconds":manifest["elapsed_seconds"]}
    write_json(output/"projection_stats.json",{"schema_version":1,"base_stat_names":BASE_NAMES,"layer_stat_names":LAYER_NAMES,"metadata":metadata,"records":records})
    write_json(output/"source_manifest.json",manifest)
    write_json(output/"check.json",check)
    if not check["passed"]:
        raise RuntimeError("Numerical checks failed; inspect check.json")
    print(f"PASS: {len(records)} records; {output}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("pred-dir","hr-dir","reference-manifest","reference-stats","out"):
        parser.add_argument(f"--{option}",required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    for source in (Path(args.pred_dir).resolve(),Path(args.hr_dir).resolve()):
        if output == source or source in output.parents:
            parser.error("--out must be outside read-only input directories")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        extract(args,output,started)
    except Exception as error:
        if not (output/"check.json").exists():
            write_json(output/"check.json",{"schema_version":1,"passed":False,"scope":"no_model_calls","error":f"{type(error).__name__}: {error}","elapsed_seconds":time.perf_counter()-started})
        print(f"FAILED: {type(error).__name__}: {error}",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
