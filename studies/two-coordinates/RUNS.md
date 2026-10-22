# Run locations

Every stage ran on the authors' GPU server. Frozen scripts keep the absolute paths they ran with (`/workspace/SE-INR/...`); to rerun, use `tools/frequency_probe.py` or adjust those paths. Large per-bin outputs stay on the server; their SHA-256 hashes are in `LARGE_ROWS.sha256`.

| Stage | Server run directory |
| --- | --- |
| `00-u-profile-reanalysis` | `/workspace/SE-INR/runs/claude-20261019-u-profile-v1` |
| `01-cell-mismatch` | `/workspace/SE-INR/runs/claude-20261019-cell-mismatch-v1` |
| `02-two-coordinates` | `/workspace/SE-INR/runs/claude-20261019-two-coordinates-v2` |
| `03-decoder-families` | `/workspace/SE-INR/runs/claude-20261019-encoder-v3` |
| `04-training-prior` | `/workspace/SE-INR/runs/claude-20261019-prior-v4` |
| `05-matched-encoder` | `/workspace/SE-INR/runs/claude-20261019-encoder-train-v5` |
| `06-paper-figures` | `/workspace/SE-INR/runs/claude-20261020-paper-figures` |
| `07-swinir` | `/workspace/SE-INR/runs/claude-20261020-swinir-v7` |
| `08-seed2` | `/workspace/SE-INR/runs/claude-20261020-seed2-v8` |
| `09-fullimage` | `/workspace/SE-INR/runs/claude-20261020-fullimage-v9` |
| `10-theory-check` | `/workspace/SE-INR/runs/claude-20261020-theory-check` |
| `11-transfer` | `/workspace/SE-INR/runs/claude-20261020-transfer-v11` |
| `12-gauss-degradation` | `/workspace/SE-INR/runs/claude-20261021-gaussdeg-v12` |
