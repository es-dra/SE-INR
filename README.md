# SE-INR / SC-INR ASISR

This repository is a research codebase derived from Rot-E ASISR for arbitrary-scale image super-resolution (ASISR). The current working direction studies scale-consistent implicit neural representations for out-of-distribution scale generalization.

## Project scope

The codebase keeps the original LIIF/LTE and rotation-equivariant Rot-E baselines, and adds scale-consistency ablations around LTE-style Fourier decoding.

Current model families:

- `liif`: standard LIIF baseline.
- `lte`: standard LTE baseline with cell-conditioned phase.
- `lte-no-cell`: LTE ablation with the cell-conditioned phase removed.
- `lte-feature-phase`: LTE ablation where phase is predicted from local feature `z` rather than cell `c`.
- `sc_inr_fixed`: SC-INR Phase 1, fixed log-polar frequency basis with analytic sinc sampling weights.
- `sc_inr_adaptive`: SC-INR Phase 2, data-driven local frequency `omega(z)` with analytic sinc sampling weights and `phi=0`.
- `liif_eq` / `lte_eq` related configs and modules: inherited Rot-E rotation-equivariant baselines.

## Main entry points

- `train.py`: training entry point.
- `test.py`: single-config evaluation entry point.
- `eval_full.py`: discrete benchmark evaluation for Set5, Set14, BSD100, and Urban100.
- `eval_continuous.py`: continuous-scale benchmark evaluation with corrected HR-crop/LR-generation alignment.
- `eval_fce.py`: function consistency error evaluation.
- `eval_phase_intervention.py`: LTE phase intervention evaluation.

## Important result files

- `results/benchmark.json`: discrete benchmark results.
- `results/continuous/set5.json`: continuous Set5 results.
- `results/continuous/bsd100.json`: continuous BSD100 results.
- `results/continuous/set14.json`: continuous Set14 results.
- `results/continuous/urban100.json`: continuous Urban100 results.
- `results/phase_intervention.json`: LTE phase intervention results.
- `results/fce.json`: most recent FCE output.

## Notes

SC-INR is currently best described as scale sampling consistency, not strict mathematical scale equivariance. Paper claims and experiment summaries should be based on the current configs, logs, and result files rather than older Rot-E README text or obsolete draft sections.
