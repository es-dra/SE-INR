#!/bin/bash
# Usage: bash launch.sh RUN_ROOT GPU_constant_s1 GPU_constant_s2 GPU_shuffle_s1 GPU_shuffle_s2
# Frozen run -> score pipeline. Each GPU is an explicitly assigned physical PCI-order ID.
set -euo pipefail
R=$(cd "$1" && pwd)
[ "$#" -eq 5 ] || { echo 'Expected run root and four distinct GPU IDs' >&2; exit 2; }
cd "$R"
sha256sum -c FROZEN.sha256 > FREEZE_VERIFIED.txt
[ ! -e STARTED.json ] || { echo 'Existing run; no retry or overwrite' >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 - "$2" "$3" "$4" "$5" <<'PY'
import json,sys,time
assert len(set(sys.argv[1:]))==4,'Use four distinct GPUs'
json.dump({'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpus':sys.argv[1:]},open('STARTED.json','w'),indent=2)
PY
mkdir logs statuses
run_slot() {
  arm=$1; seed=$2; gpu=$3; label=${arm}_s${seed}
  set +e
  timeout --signal=TERM --kill-after=120s 24h bash "$R/run_slot.sh" "$R" "$arm" "$seed" "$gpu" > "$R/logs/slot_${label}.log" 2>&1
  result=$?
  echo "$result" > "$R/statuses/${label}.exit"
  return "$result"
}
run_slot constant 1 "$2" & p1=$!
run_slot constant 2 "$3" & p2=$!
run_slot shuffle 1 "$4" & p3=$!
run_slot shuffle 2 "$5" & p4=$!
failed=0
for pid in "$p1" "$p2" "$p3" "$p4"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then echo 'At least one slot failed; preserve failure, do not score incomplete results.' > PIPELINE_FAILED.txt; exit 1; fi
sha256sum -c FROZEN.sha256 > FREEZE_VERIFIED_BEFORE_SCORE.txt
python3 "$R/score.py" --run-root "$R" > "$R/logs/score.log" 2>&1
printf 'complete\n' > COMPLETE.txt
