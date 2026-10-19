#!/bin/bash
cd /workspace/SE-INR/claude-20261005-e2r; export NTH=4
cat jobs_rerun.txt | xargs -P 7 -L 1 bash -c 'python3 e2r.py --task $0 --img $1 --out /workspace/SE-INR/claude-20261005-e2r/out/e2r_$0_$1.json > /workspace/SE-INR/claude-20261005-e2r/out/e2r_$0_$1.log 2>&1'
echo done > out/RERUN_DONE
