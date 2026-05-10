# Experiment: SC-INR-NoSinc

## Question

Does the final-candidate SC-INR benefit from the analytic sinc observation
response, beyond signed feature-conditioned omega and feature-conditioned phase?

## Hypothesis

If the analytic response is a meaningful part of decoder-side sampling
consistency, removing it should weaken OOD scale robustness and/or same-LR
cross-scale consistency relative to `SC-INR`.

## Frozen Protocol

- Models: `SC-INR`, `SC-INR-NoSinc`; report `LTE`, `SC-INR-NoPhi`, and
  `SC-INR-NoPhi-Signed` as controls when evaluating.
- Dataset split: DIV2K train/valid as in `configs/train-div2k/train-sc-inr.yaml`.
- Scales: training `scale_max: 4`; benchmark ID x2/x3/x4 and OOD x6/x8/x12/x16/x24/x30.
- Seeds: start with seed1 only; do not mix seed1 with multi-seed conclusions.
- Checkpoint rule: use `epoch-best.pth` under `artifacts/checkpoints/seed1/sc-inr-nosinc`.
- Metrics: PSNR, auxiliary PSNR-Y/SSIM-Y/texture/highpass RMSE, same-LR x8/x16/x30 -> x4 consistency.
- Baseline: final-candidate `SC-INR` with identical signed omega and `phi(z)`, but with `use_sinc_response: true`.
- Forbidden post-hoc changes: do not tune omega bounds, phase settings, data protocol, or checkpoint selection after seeing NoSinc results.

## Commands

- Smoke:
  `python -m py_compile src/models/sc_inr_adaptive.py`
- Train:
  `CUDA_VISIBLE_DEVICES=<gpu> python entrypoints/train.py --config configs/train-div2k/train-sc-inr-nosinc.yaml --name sc-inr-nosinc --saveFolder artifacts/checkpoints/seed1 --seed 1`
- Benchmark:
  `python entrypoints/eval_full.py --device <gpu> --save_root artifacts/checkpoints/seed1 --models SC-INR-NoSinc --output artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json`
- Auxiliary:
  `python scripts/analysis/evaluate_seed1_aux_metrics.py --models LTE,SC-INR-NoPhi,SC-INR-NoPhi-Signed,SC-INR,SC-INR-NoSinc --datasets bsd100,urban100 --scales 4,8,16,30 --consistency_pairs '8,4;16,4;30,4' --max_images 10 --out artifacts/derived/analysis/sc_inr_nosinc_aux_metrics_seed1 --skip_visuals`

## Artifacts

- Config: `configs/train-div2k/train-sc-inr-nosinc.yaml`
- Checkpoints: `artifacts/checkpoints/seed1/sc-inr-nosinc/`
- Raw results: `artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json`
- Derived analysis: `artifacts/derived/analysis/sc_inr_nosinc_aux_metrics_seed1/`
- Logs: `artifacts/checkpoints/seed1/sc-inr-nosinc/log.txt`

## Result

Observation: seed1 benchmark complete.

- Raw result: `artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json`
- Derived summary: `artifacts/derived/analysis/benchmark_progress_2026-05-10/seed1_signed_phiz_nosinc_summary.csv`
- `SC-INR-NoSinc`: ID `30.9790`, OOD `22.7299`, All `25.4796`
- `SC-INR-NoSinc - LTE`: ID `-0.0659`, OOD `+0.0382`, All `+0.0035`
- `SC-INR-NoSinc - SC-INR`: ID `-0.1397`, OOD `-0.0467`, All `-0.0777`
- `SC-INR-NoSinc - SC-INR-NoPhi`: ID `-0.0724`, OOD `-0.0172`, All `-0.0356`

Conclusion strength: `single-seed benchmark evidence plus seed1 auxiliary diagnostic`

Initial interpretation: removing analytic sinc response weakens the final
candidate relative to full `SC-INR`, especially on ID PSNR and all-scale
average. OOD remains slightly above `LTE`, so this benchmark alone supports a
contribution from sinc but does not prove sinc is the unique source of the
method's gains.

## Auxiliary Result

Observation: seed1 auxiliary metrics complete.

- Derived analysis: `artifacts/derived/analysis/sc_inr_nosinc_aux_metrics_seed1/`
- Protocol: BSD100 / Urban100 sorted 前 10 张；x4/x8/x16/x30 quality；
  same-LR x8/x16/x30 -> x4 consistency；models include `LTE`,
  `SC-INR-NoPhi`, `SC-INR-NoPhi-Signed`, `SC-INR`, `SC-INR-NoSinc`.
- Coverage: 480 quality rows and 300 consistency rows; shared-model summary rows
  exactly match `sc_inr_final_aux_metrics_seed1`.
- OOD PSNR-Y all average over BSD100/Urban100 x8/x16/x30:
  `SC-INR-NoSinc` 21.7284 vs `SC-INR` 21.7218 (`+0.0066 dB`).
- OOD texture RMSE-Y all average:
  `SC-INR-NoSinc` 0.152998 vs `SC-INR` 0.153208 (`-0.000210`, lower is better).
- Same-LR consistency PSNR-Y all average:
  `SC-INR-NoSinc` 66.8555 vs `SC-INR` 49.6275 (`+17.2280 dB`).

Interpretation: auxiliary consistency does not show NoSinc breaking same-LR
self-consistency. It shows the opposite, likely because disabling sinc removes
cell-dependent observation response and makes the decoder closer to a
cell-independent point function. This is a diagnostic limitation: same-LR
consistency must be interpreted together with fidelity and response diagnostics.
The full benchmark still suggests sinc contributes to final-candidate PSNR.

## Caveats

This ablation can test whether removing sinc hurts under the same training
protocol. It cannot by itself prove a unique causal mechanism if optimization
variance or phase/omega compensation explains the result.

The auxiliary result weakens the naive mechanism story: under the current metric,
removing sinc increases same-LR self-consistency rather than decreasing it. Paper
claims should therefore frame sinc support through full benchmark/fidelity
evidence and treat the consistency result as a diagnostic caveat.

## Response/Omega Diagnostic

Observation: response/omega diagnostics complete.

- Derived analysis: `artifacts/derived/analysis/response_omega_diagnostics_2026-05-10/`
- Protocol: response distribution on BSD100 / Urban100 sorted 前 5 张; cell-only
  curve on BSD100 / Urban100 sorted 前 5 张, same x4 LR and same coordinates,
  changing only cell scale.
- `SC-INR-NoSinc` active attenuation mean is `0` at x4/x8/x16/x30 because the
  model actually uses `W=1`.
- `SC-INR-NoSinc` cell-only RMSE-Y vs x4 is exactly `0` at all tested scales.
- Full `SC-INR` active attenuation mean is nonzero:
  x4 `0.040495`, x8 `0.010570`, x16 `0.002672`, x30 `0.000762`.
- Full `SC-INR` cell-only RMSE-Y vs x4 at x8/x16/x30 averages `0.003255`,
  below `LTE` (`0.007943`) but above `SC-INR-NoSinc` (`0`).

Interpretation: NoSinc's high same-LR consistency is explained by zero active
cell response, not by a better footprint observation model. Full `SC-INR` keeps
a finite analytic cell response while avoiding LTE's larger learned cell-phase
sensitivity.
