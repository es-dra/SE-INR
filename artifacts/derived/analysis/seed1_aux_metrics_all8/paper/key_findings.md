# Seed-1 Auxiliary Metric Findings

Inputs: `quality_summary.csv` and `consistency_summary.csv` from `seed1_aux_metrics_all8`.
Quality OOD means average x8, x16, and x30. Consistency means average x8/x16/x30 -> x4.

## SC-INR-NoPhi vs. LTE

- BSD100 x4 PSNR delta: +0.004 dB.
- BSD100 OOD PSNR delta: +0.056 dB; OOD SSIM delta: +0.00191.
- BSD100 OOD texture RMSE delta: -0.00095.
- BSD100 consistency PSNR delta: +8.61 dB; texture consistency RMSE delta: -0.00590.
- Urban100 x4 PSNR delta: +0.031 dB.
- Urban100 OOD PSNR delta: +0.052 dB; OOD SSIM delta: +0.00101.
- Urban100 OOD texture RMSE delta: -0.00094.
- Urban100 consistency PSNR delta: +7.77 dB; texture consistency RMSE delta: -0.01047.

## Interpretation Caveat

LTE-NoCellPhase and LTE-PhaseZ obtain much higher cross-scale consistency because their output is weakly conditioned on cell size. They should be treated as diagnostic controls rather than better ASISR models unless reconstruction quality and texture errors are considered jointly.

## Paper Use

- Use `quality_psnr_table.tex` for the compact PSNR table.
- Use `quality_ssim_texture_table.tex` for perceptual/texture support.
- Use `consistency_table.tex` for the same-LR cross-scale observation consistency metric.
- Use `fig_quality_consistency_tradeoff.pdf` to show that SC-INR-NoPhi improves consistency while preserving quality better than purely removing cell response.
