#!/usr/bin/env bash
# One GPU process; no scoring before the complete selection cohort is frozen.
set -euo pipefail
TASK=/workspace/SE-INR/codex-20261008-paper-continuation
CODE=$TASK/guidance
ROOT=$TASK/guidance-formal-v4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-675adc8d-a6ea-1c8b-f833-f8ae8a87cdaa
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH=$CODE
python3 - "$TASK" "$CODE" <<'PY'
import hashlib,json,pathlib,sys
base,code=map(pathlib.Path,sys.argv[1:])
hashes={key:hashlib.sha256((code/file).read_bytes()).hexdigest() for key,file in
        [('core_sha256','core.py'),('experiment_sha256','experiment.py'),
         ('protocol_sha256','protocol.json'),('summary_sha256','summarize.py')]}
for mode,marker in [('full','NUMERICAL_SMOKE_COMPLETE'),('mini','COMPLETE')]:
    root=base/f'guidance-development-{mode}-v4'
    assert (root/marker).is_file(),f'Development incomplete: {root}'
    frozen=json.loads((root/'frozen.json').read_text())
    assert all(frozen[k]==v for k,v in hashes.items()),'Development source changed'
    paths=sorted((root/'selections').glob('*.json'))
    assert len(paths)==(1 if mode=='full' else 6),'Development case coverage changed'
    for path in paths:
        state=json.loads(path.read_text())
        assert state['status']=='complete' and not state['failures'],str(path)
        assert len(state['scores'])==len(state['cv'])==8,str(path)
        records=[n for section in ['scores','cv'] for arm in state[section].values()
                 for n in arm['numerics']]
        records += [arm['numerics'] for arm in state['reconstructions'].values()]
        assert all(n['converged'] and n['true_relative_residual']<1e-5 for n in records),str(path)
assert json.loads((code/'CPU检查-服务器独立.json').read_text())['status']=='pass'
print('Development convergence and frozen sources verified; target selection may start.',flush=True)
PY
run() {
  nice -n 10 python3 -u -c 'import torch,runpy;torch.cuda.set_per_process_memory_fraction(.15);runpy.run_path("/workspace/SE-INR/codex-20261008-paper-continuation/guidance/experiment.py",run_name="__main__")' "$@"
}
run calibrate --output "$ROOT"
for image in 0813 0814 0815 0816 0817 0818 0819 0820 0821 0822 0823 0824; do
  for noise in .005 .01 .02; do
    run prepare --output "$ROOT" --image "$image" --noise "$noise"
    run select --output "$ROOT" --image "$image" --noise "$noise"
  done
done
run freeze-selections --output "$ROOT" --split target
for image in 0813 0814 0815 0816 0817 0818 0819 0820 0821 0822 0823 0824; do
  for noise in .005 .01 .02; do
    run score --output "$ROOT" --image "$image" --noise "$noise"
  done
done
python3 "$CODE/summarize.py" --output "$ROOT" --split target
date -u > "$ROOT/COMPLETE"
