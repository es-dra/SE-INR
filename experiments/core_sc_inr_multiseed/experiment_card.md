# Experiment: Core SC-INR-NoPhi Multi-Seed Benchmark

## Question

Does the no-phase SC-INR variant improve OOD scale robustness over LTE under
the same benchmark protocol across multiple seeds?

## Frozen Protocol

- Models: `LIIF`, `LTE`, `SC-INR-NoPhi` (raw key `SC-INR`)
- Seeds: 1, 2, 3
- Datasets: Set5, Set14, BSD100, Urban100
- Scales: x2, x3, x4, x6, x8, x12, x16, x24, x30
- ID: x2/x3/x4
- OOD: x6/x8/x12/x16/x24/x30
- Checkpoint rule: `epoch-best.pth`
- Raw results:
  - `artifacts/raw_results/seed1/benchmark_signed_phiz.json`
  - `artifacts/raw_results/seed2/benchmark.json`
  - `artifacts/raw_results/seed3/benchmark.json`

## Result

Current derived summary:

- `artifacts/derived/analysis/benchmark_progress_2026-05-09/multiseed_core_summary.csv`

SC-INR-NoPhi relative to LTE:

- ID: `-0.0040 dB`
- OOD: `+0.0507 dB`
- All: `+0.0325 dB`

## Claim Boundary

This supports stable small OOD PSNR improvement for the no-phase variant. It
does not support strict scale equivariance, broad SOTA claims, or final-candidate
`SC-INR` multi-seed superiority.
