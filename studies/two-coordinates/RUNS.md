# Run locations

Original model runs used the authors' GPU server. Stages 15–16 perform CPU analysis of saved outputs only. Frozen scripts keep the absolute paths they ran with (`/workspace/SE-INR/...`); to rerun, use `tools/frequency_probe.py` or adjust those paths. Large per-bin outputs stay on the server; their SHA-256 hashes are in `LARGE_ROWS.sha256`.

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
| `13-benchmark` | `/workspace/SE-INR/runs/claude-20261022-benchmark-v13` |
| `14-qualitative` | `/workspace/SE-INR/runs/claude-20261023-qualitative-v14` |
| `15-energy-filter-decomposition` (offline saved-output analysis) | `/workspace/SE-INR/runs/codex-20261023-energy-filter-v1` (CPU extraction); final decomposition is in the repository's `15-energy-filter-decomposition/final/` |
| 16-residual-structure (offline saved-output analysis and paper revision) | /workspace/SE-INR/runs/codex-20261023-residual-structure-v1; projection extraction: 213.58 seconds wall time, 336 existing files; response extraction: 2.41 seconds; no model calls. Final local outputs: localization-v2, projection-summary-v3, step-controls-v1. LaTeX builds isolated under this new run root. |
