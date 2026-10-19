#!/bin/bash
set -euo pipefail
cd /workspace/SE-INR/codex-20261011-readout-probe-v4
python - <<'META'
import json,time,pathlib,hashlib
p=pathlib.Path('formal-launch.json'); assert not p.exists()
t=time.time(); p.write_text(json.dumps(dict(start_epoch=t,absolute_deadline_epoch=t+2400,budget_seconds=2400,source_sha256=hashlib.sha256(pathlib.Path('experiment.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(pathlib.Path('protocol.json').read_bytes()).hexdigest(),device='cpu',threads=4,nice=10),indent=2))
META
TIMEFORMAT='real_seconds=%R user_seconds=%U system_seconds=%S'
{ time timeout 2400s env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 python -X faulthandler experiment.py --launch-record formal-launch.json --out /workspace/SE-INR/codex-20261011-readout-formal-v4; } 2> formal-fullprocess-cost.txt
