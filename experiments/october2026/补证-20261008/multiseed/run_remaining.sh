#!/usr/bin/env bash
set -euo pipefail
TASK=/workspace/SE-INR/codex-20261008-paper-continuation
SOURCE=/workspace/SE-INR/claude-20261005-L12
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-675adc8d-a6ea-1c8b-f833-f8ae8a87cdaa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TASK/logs"
date -u > "$TASK/multiseed/STARTED"
sha256sum "$SOURCE/l2s.py" "$SOURCE/common.py" "$SOURCE/ana_l2s.py" > "$TASK/multiseed/source.sha256"
run_pair() {
  local seed="$1" pair="$2" output="$3"
  timeout 3600 nice -n 10 python3 -u -c 'import torch,runpy; torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(0.15); runpy.run_path("/workspace/SE-INR/claude-20261005-L12/l2s.py",run_name="__main__")' --seed "$seed" --pair "$pair" --outdir "$TASK/multiseed/$output" > "$TASK/logs/$output.log" 2>&1
  python3 "$SOURCE/ana_l2s.py" "$TASK/multiseed/$output" "$pair" > "$TASK/logs/analysis_$output.log" 2>&1
  python3 - "$TASK/multiseed/$output/l2s_result.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['H']['H2z']['ok'], 'Zero control failed; interpretation must stop'
assert len(d['L2'])==1
assert len(next(iter(d['L2'].values()))['rows'])==36
PY
}
run_pair 2 sc-inr out_s2_sc
run_pair 3 lte out_s3_lte
date -u > "$TASK/multiseed/COMPLETE"
