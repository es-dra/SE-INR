# Footprint extrapolation, stage 01: inference on existing checkpoints

Question, conditions, frozen predictions and interpretations: [`protocol.json`](protocol.json). No training. Uses the 1000-epoch EDSR-baseline LIIF, LTE and SC-INR checkpoints (seeds 1 and 2; seed 3 if present) and the 246 full images of DIV2K 0869–0900, Set14, BSD100 and Urban100.

## Run (server)

```bash
cd /workspace/SE-INR/SE-INR-current && git pull
python3 studies/footprint/01-inference/freeze.py --run-root /workspace/SE-INR/runs/claude-20261101-footprint-v1
cd /workspace/SE-INR/runs/claude-20261101-footprint-v1 && nohup bash launch.sh 1,2 > launch.log 2>&1 &
```

Replace `1,2` with free GPUs (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). Six shards of about 40 full-image forward passes per image; expect a few hours on two GPUs. Results: `score/summary.json`. Copy back into this folder: `FROZEN.sha256`, `input_manifest.json`, `RUN_STARTED.json`, `INFERENCE_COMPLETE.json`, `outputs/*/COMPLETE.json`, `outputs/*/metadata.json`, `score/`. Records stay on the server.

## Self-test (any machine, CPU)

```bash
python3 studies/footprint/01-inference/run.py --smoke /tmp/footprint-smoke
```
