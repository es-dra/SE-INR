#!/usr/bin/env python3
"""Full benchmark: 4 models × 4 benchmarks × multiple scales + cross-scale evaluation."""

import subprocess
import json
import os
import tempfile
import shutil

BASE = "/workspace/SE-INR/Equivariant-ASISR"
os.chdir(BASE)

MODELS = [
    ("save/edsr-baseline-liif/epoch-best.pth",    "LIIF_baseline"),
    ("save/edsr-baseline-liif-EQ/epoch-best.pth", "LIIF_EQ"),
    ("save/se-inr-s0/epoch-best.pth",              "SE-INR_S0"),
    ("save/se-inr-s1/epoch-best.pth",              "SE-INR_S1"),
]

BENCHMARKS = ["Set5", "Set14", "BSD100", "Urban100"]
STD_SCALES = [2, 3, 4]
CROSS_SCALES = [6, 8, 12, 18, 24, 30]

results = {}

def run_eval(model_path, bench, scale):
    """Run test.py and extract PSNR result."""
    # Build config on the fly
    hr_path = f"/workspace/SE-INR/Data/{bench}/HR"
    if not os.path.exists(hr_path):
        return None

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

    try:
        result = subprocess.run(
            ["python", "test.py", "--config", config_file, "--model", model_path, "--device", "cuda"],
            capture_output=True, text=True, timeout=300
        )
        for line in result.stdout.split('\n') + result.stderr.split('\n'):
            if 'result:' in line.lower() or 'psnr' in line.lower():
                # Try to extract numeric PSNR
                import re
                nums = re.findall(r'\d+\.\d+', line)
                for num in nums:
                    val = float(num)
                    if 20 < val < 50:  # sanity check
                        return val
        return None
    except Exception as e:
        return None
    finally:
        os.unlink(config_file)

# --- Standard benchmark (×2/×3/×4) ---
print("=" * 70)
print("STANDARD BENCHMARK: 4 models × 4 benchmarks × 3 scales")
print("=" * 70)

results["standard"] = {}
for model_path, model_name in MODELS:
    results["standard"][model_name] = {"by_benchmark": {}}
    print(f"\n>>> {model_name}")
    for bench in BENCHMARKS:
        results["standard"][model_name]["by_benchmark"][bench] = {}
        for scale in STD_SCALES:
            print(f"  {bench} x{scale}...", end=" ", flush=True)
            val = run_eval(model_path, bench, scale)
            if val is not None:
                results["standard"][model_name]["by_benchmark"][bench][f"x{scale}"] = round(val, 4)
                print(f"{val:.4f} dB")
            else:
                results["standard"][model_name]["by_benchmark"][bench][f"x{scale}"] = None
                print("FAILED")

# --- Cross-scale evaluation (×6~×30) on Set5 only ---
print("\n" + "=" * 70)
print("CROSS-SCALE EVALUATION: 4 models × Set5 × 6 scales (×6~×30)")
print("=" * 70)

results["cross_scale"] = {}
for model_path, model_name in MODELS:
    results["cross_scale"][model_name] = {}
    print(f"\n>>> {model_name}")
    for scale in CROSS_SCALES:
        print(f"  Set5 x{scale}...", end=" ", flush=True)
        val = run_eval(model_path, "Set5", scale)
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
