# Diagnostics

本目录只保存 appendix / diagnostic 层级的派生证据。它们可以帮助解释机制、
负控和 caveat，但不能替代 `artifacts/derived/benchmarks/` 中的主 benchmark。

## 当前内容

- `seed1_aux_metrics_all8/`：旧 no-phase 主线的 seed1 quality / consistency 辅助证据。
- `sc_inr_final_aux_metrics_seed1/`：最终候选 `SC-INR` 的 seed1 辅助指标。
- `sc_inr_nosinc_aux_metrics_seed1/`：NoSinc 负控和 consistency caveat。
- `response_omega_diagnostics_2026-05-10/`：response/omega 机制诊断。
- `effective_amplitude_2026-05-12/`：`q_coef(z) * W(omega(z), c)` 的 feature-level
  effective amplitude 诊断。
- `mechanism_suite_2026-05-12/`：6.1-6.6 与 SC-INR-EQ 的 active 摘要；完整探索产物在 legacy。

## 使用边界

论文引用应先经过 `artifacts/derived/evidence/README_zh.md` 和
`paper/ARTIFACTS_ALLOWED.md`。本目录中的结果默认是 seed1、small-sample 或
diagnostic，不得写成 multi-seed 主结论。
