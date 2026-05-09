# SE-INR / SC-INR ASISR

This is a paper-grade research workspace for arbitrary-scale image
super-resolution (ASISR), derived from Rot-E ASISR and focused on
sampling-consistent Fourier implicit decoding.

The current method family is best described as **scale sampling consistency** or
**scale-decoupled observation**, not strict mathematical scale equivariance.

## Current Research Question

LTE predicts Fourier frequency/coefficient from image features, but its phase is
conditioned directly on output cell/scale. SC-INR moves scale out of the learned
phase shortcut and lets output cell affect the observation through an analytic
sinc response. `SC-INR+PhiZ` further predicts phase from features only:

- content Fourier basis: `coef(z)`, `omega(z)`, optionally `phi(z)`;
- output footprint: analytic `sinc(omega * cell / 2)`;
- forbidden shortcut: learned `phase(cell)`.

## Directory Map

| Path | Purpose |
| --- | --- |
| `src/` | Canonical source code: models, datasets, utilities. |
| `entrypoints/` | Canonical CLIs for train/eval/diagnostics. Root entrypoint files remain as symlinks for compatibility. |
| `configs/` | Training/evaluation configs plus `configs/registry/` model/protocol aliases. |
| `scripts/analysis/` | Analysis and diagnostic scripts. |
| `scripts/paper/` | Scripts that prepare paper-facing tables/figures. |
| `scripts/viz/` | Qualitative visualization tools. |
| `artifacts/checkpoints/` | Checkpoints, saved configs, training logs. Ignored by git. |
| `artifacts/raw_results/` | Formal raw metric JSON files. |
| `artifacts/derived/` | Derived CSV/TEX/figures and paper candidates. |
| `artifacts/smoke/` | Smoke/debug outputs; do not cite. |
| `artifacts/legacy/` | Historical outputs retained for traceability. |
| `artifacts/results/` | Compatibility tree for old `results/...` paths. |
| `experiments/` | Experiment cards and manifests. |
| `paper/` | Claims/evidence ledger and paper-facing workspace. |
| `docs/` | Project documentation and archived historical reports. |
| `memory/` | Chronological daily research memory. |

See [MANIFEST.md](MANIFEST.md) for the full project contract.

## Compatibility Paths

The repo was reorganized without breaking common old commands. These root paths
are symlinks:

- `train.py` -> `entrypoints/train.py`
- `eval_full.py` -> `entrypoints/eval_full.py`
- `models` -> `src/models`
- `datasets` -> `src/datasets`
- `save` -> `artifacts/checkpoints/seed1`
- `save-seeds` -> `artifacts/checkpoints/seeds`
- `results` -> `artifacts/results`
- `logs` -> `artifacts/logs`

New scripts should prefer canonical paths; old paths are compatibility only.

## Main Entry Points

```bash
# Train
python train.py --config configs/train-div2k/train-sc-inr-signed-phiz.yaml \
  --name sc-inr-phiz --saveFolder ./save --device 1 --seed 1

# Discrete benchmark
python eval_full.py --device 1 --save_root ./save \
  --models 'SC-INR+PhiZ' \
  --output results/benchmark_seed1_with_signed_phiz.json --skip_existing

# Qualitative candidates
python scripts/prepare_qualitative_figure.py \
  --dataset urban100 --image img_004.png --scale 8 \
  --models Bicubic,LIIF,LTE,SC-INR,SC-INR+PhiZ \
  --auto_delta_crop --target_model SC-INR+PhiZ --baseline_model LTE
```

## Model Registry

Paper-facing model aliases live in
[configs/registry/models.yaml](configs/registry/models.yaml). Important aliases:

- `SC-INR`: display name for the previous `SC-INR-Adaptive` checkpoint.
- `SC-INR-Signed`: signed bounded omega diagnostic variant.
- `SC-INR+PhiZ`: feature-conditioned phase candidate.

Do not rename checkpoint directories just to match display names. Use the
registry to map display names to configs, registry keys, and checkpoint paths.

## Current Evidence Entry Points

- Core multi-seed benchmark:
  [experiments/core_sc_inr_multiseed/experiment_card.md](experiments/core_sc_inr_multiseed/experiment_card.md)
- `SC-INR+PhiZ` candidate:
  [experiments/phiz/experiment_card.md](experiments/phiz/experiment_card.md)
- Claim/evidence ledger:
  [paper/claims_evidence_matrix.md](paper/claims_evidence_matrix.md)
- Paper artifact whitelist:
  [paper/ARTIFACTS_ALLOWED.md](paper/ARTIFACTS_ALLOWED.md)
- Current benchmark derived summary:
  `artifacts/derived/analysis/benchmark_progress_2026-05-09/`
- Seed1 auxiliary consistency evidence:
  `artifacts/derived/analysis/seed1_aux_metrics_all8/`
- Qualitative PhiZ candidates:
  `artifacts/derived/paper_candidates/qualitative_phiz_candidates/`

## Data

Training/evaluation configs expect the project data root at `../Data` by
default. Several evaluation entrypoints also honor:

```bash
export SEINR_DATA_ROOT=/workspace/SE-INR/Data
```

The repo-local `Data` path is a compatibility symlink for inherited benchmark
files and is not the authoritative dataset store.

## Claim Boundary

Allowed:

- "SC-INR improves decoder-side sampling consistency."
- "SC-INR yields stable small OOD PSNR gains over LTE under the current
  benchmark protocol."
- "`SC-INR+PhiZ` is a promising seed1 candidate with stronger PSNR and selected
  qualitative improvements."

Forbidden until further evidence:

- "SC-INR is strictly scale-equivariant."
- "`SC-INR+PhiZ` is the final multi-seed winner."
- "The gain is caused by sinc" without a w/o-sinc ablation.

## Historical Material

Historical reports are kept under `docs/archive/`. They are useful for research
traceability but may contain stale conclusions superseded by later evaluation.
