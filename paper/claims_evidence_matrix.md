# Claims And Evidence Matrix

This ledger keeps paper claims proportional to available evidence.

| Claim | Evidence | Strength | Allowed Wording | Forbidden Overclaim |
| --- | --- | --- | --- | --- |
| SC-INR improves OOD scale robustness over LTE. | 3-seed core benchmark in `artifacts/derived/analysis/benchmark_progress_2026-05-09/multiseed_core_summary.csv`: OOD `+0.0507 dB` vs LTE. | Verified for current protocol/core models. | "SC-INR yields a stable small OOD PSNR improvement over LTE under our benchmark protocol." | "SC-INR is strictly scale-equivariant" or "large SOTA gain". |
| SC-INR improves same-LR cross-scale observation consistency. | `artifacts/derived/analysis/seed1_aux_metrics_all8/paper/consistency_table.tex`: BSD100 `+8.61 dB`, Urban100 `+7.77 dB` vs LTE. | Strong seed1 mechanism evidence. | "Decoder-side sampling consistency substantially improves same-LR cross-scale consistency." | "The whole network is scale equivariant." |
| SC-INR+PhiZ is a promising stronger candidate. | Seed1 benchmark in `benchmark_signed_phiz.json`; derived summary shows All `+0.0812 dB` vs LTE and `+0.0421 dB` vs SC-INR. | Preliminary. | "SC-INR+PhiZ is a promising seed1 candidate with stronger PSNR." | "SC-INR+PhiZ is the final best model" before multi-seed/aux metrics. |
| PhiZ improves visible local structure in selected examples. | User-confirmed candidate figures: Urban100 img012 x8 and img004 x8. | Qualitative candidate evidence. | "Selected examples show clearer local structure." | "Average visual quality is better" without broader metrics/distribution. |
| Benefits come from analytic sinc response. | Existing consistency evidence supports the mechanism direction. | Incomplete. | "The analytic response is consistent with improved scale sampling behavior." | "The gain is caused by sinc" until w/o sinc ablation exists. |
