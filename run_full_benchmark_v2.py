#!/usr/bin/env python3
"""Full benchmark: 4 models × 4 benchmarks × 3 scales + cross-scale (Set5 only)."""

import subprocess
import json
import os
import tempfile

BASE = "/workspace/SE-INR/Equivariant-ASISR"
os.chdir(BASE)

MODELS = [
    ("save/edsr-baseline-liif/epoch-best.pth",    "LIIF_baseline"),
    ("save/edsr-baseline-liif-EQ/epoch-best.pth", "LIIF_EQ"),
    ("save/se-inr-s0/epoch-best.pth",              "SE-INR_S0"),
    ("save/se-inr-s1/epoch-best.pth",              "SE-INR_S1"),
]

BENCHMARKS = ["Set5", "Set14", "Urban100"]  # Skip BSD100 for ×2/×3/×4 (use stored configs)
BENCHMARKS_ALL = ["Set5", "Set14", "BSD100", "Urban100"]  # Full list
STD_SCALES = [2, 3, 4]
CROSS_SCALES = [6, 8, 12, 18, 24, 30]

results = {}

def get_timeout(bench, scale):
    """Dynamic timeout based on dataset size and scale."""
    base = {"Set5": 60, "Set14": 120, "BSD100": 600, "Urban100": 180}[bench]
    # Higher scales = more output pixels = slower
    return int(base * (scale / 2))

def run_eval(model_path, bench, scale, use_existing_config=False):
    """Run test.py and extract PSNR result."""
    hr_path = f"/workspace/SE-INR/Data/{bench}/HR"
    if not os.path.exists(hr_path):
        return None

    # Try existing config first (for std scales)
    config_file = None
    if use_existing_config:
        existing = f"configs/test/test-{bench.lower()}-{scale}.yaml"
        if os.path.exists(existing):
            config_file = existing

    if config_file is None:
        config_content = f"""test_dataset:
  dataset:
    name: image-folder
    args:
      root_path: {hr_path}
  wrapper:
    name: sr-implicit-downsampled
    args:
      scale_min: {scale}
      scale_max: {scale}
  batch_size: 1

data_norm:
  inp: {{sub: [0.5], div: [0.5]}}
  gt: {{sub: [0.5], div: [0.5]}}

eval_type: benchmark-{scale}
eval_bsize: 50000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_file = f.name
        temp_created = True
    else:
        temp_created = False

    try:
        timeout = get_timeout(bench, scale)
        result = subprocess.run(
            ["python", "test.py", "--config", config_file, "--model", model_path, "--device", "cuda"],
            capture_output=True, text=True, timeout=timeout
        )
        # Parse "result: XX.XXXX" from stdout (last occurrence)
        import re
        all_text = result.stdout + result.stderr
        matches = re.findall(r'result:\s*(\d+\.\d+)', all_text)
        if matches:
            return float(matches[-1])
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None
    finally:
        if temp_created and os.path.exists(config_file):
            os.unlink(config_file)

# --- Standard benchmark (×2/×3/×4) ---
print("=" * 70)
print("STANDARD BENCHMARK: 4 models × 4 benchmarks × 3 scales (x2/x3/x4)")
print("=" * 70)

results["standard"] = {}
for model_path, model_name in MODELS:
    results["standard"][model_name] = {"by_benchmark": {}}
    print(f"\n>>> {model_name}")
    for bench in BENCHMARKS_ALL:
        results["standard"][model_name]["by_benchmark"][bench] = {}
        for scale in STD_SCALES:
            print(f"  {bench} x{scale}...", end=" ", flush=True)
            val = run_eval(model_path, bench, scale, use_existing_config=True)
            if val is not None:
                results["standard"][model_name]["by_benchmark"][bench][f"x{scale}"] = round(val, 4)
                print(f"{val:.4f} dB")
            else:
                results["standard"][model_name]["by_benchmark"][bench][f"x{scale}"] = None
                print("FAILED")

# --- Cross-scale evaluation (×6~×30) on Set5 ---
print("\n" + "=" * 70)
print("CROSS-SCALE EVALUATION: 4 models × Set5 × 6 scales (x6~x30)")
print("=" * 70)

results["cross_scale"] = {}
for model_path, model_name in MODELS:
    results["cross_scale"][model_name] = {}
    print(f"\n>>> {model_name}")
    for scale in CROSS_SCALES:
        print(f"  Set5 x{scale}...", end=" ", flush=True)
        val = run_eval(model_path, "Set5", scale, use_existing_config=False)
        if val is not None:
            results["cross_scale"][model_name][f"x{scale}"] = round(val, 4)
            print(f"{val:.4f} dB")
        else:
            results["cross_scale"][model_name][f"x{scale}"] = None
            print("FAILED")

# Save results
out_path = "/workspace/SE-INR/Equivariant-ASISR/benchmark_full_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("\n" + json.dumps(results, indent=2))
