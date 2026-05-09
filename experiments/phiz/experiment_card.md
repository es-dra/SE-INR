# Experiment: SC-INR+PhiZ

## Question

Does feature-conditioned phase improve SC-INR reconstruction quality without
reintroducing LTE-style cell-conditioned phase extrapolation?

## Current Evidence

- Raw seed1 benchmark: `artifacts/raw_results/seed1/benchmark_signed_phiz.json`
- Derived seed1 summary: `artifacts/derived/analysis/benchmark_progress_2026-05-09/seed1_signed_phiz_summary.csv`
- Qualitative candidates: `artifacts/derived/paper_candidates/qualitative_phiz_candidates/`

## Current Result

Seed1 PSNR benchmark:

- `SC-INR+PhiZ - LTE`: ID `+0.0738 dB`, OOD `+0.0849 dB`, All `+0.0812 dB`
- `SC-INR+PhiZ - SC-INR`: ID `+0.0673 dB`, OOD `+0.0295 dB`, All `+0.0421 dB`

User-confirmed qualitative candidates:

- `urban100_img012_x8_delta_phiz_vs_lte.png`
- `urban100_img004_x8_delta_phiz_vs_lte.png`

## Caveats

- Single seed only.
- Auxiliary consistency/texture metrics are not yet complete for PhiZ.
- Auto-delta qualitative crops are candidate evidence, not average visual quality proof.

## Next Gates

- Run auxiliary metrics for `SC-INR+PhiZ`.
- Add `SC-INR+PhiZ w/o sinc` or equivalent sinc ablation.
- Run multi-seed if auxiliary metrics do not reveal consistency regression.
