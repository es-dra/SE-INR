# 辅助证据入口

本目录只作为辅助证据索引，不存放新的 raw benchmark。论文主 benchmark 入口见
`artifacts/derived/benchmarks/`。

## 当前 canonical 辅助证据

| 证据类型 | 入口 | 状态 | 允许用途 |
| --- | --- | --- | --- |
| seed1 SC-INR-NoPhi consistency/quality | `../diagnostics/seed1_aux_metrics_all8/paper/` | cite_ok/context | 支持 no-phase 主线的 same-LR consistency 与辅助质量指标。 |
| final SC-INR seed1 auxiliary | `../diagnostics/sc_inr_final_aux_metrics_seed1/` | preliminary | 说明 final candidate 保留对 LTE 的 strong consistency，但不是 consistency 最强变体。 |
| NoSinc caveat | `../diagnostics/sc_inr_nosinc_aux_metrics_seed1/` | diagnostic | 说明 same-LR consistency 会奖励 cell-insensitive decoder，不能单独证明 sinc。 |
| response/omega diagnostics | `../diagnostics/response_omega_diagnostics_2026-05-10/` | diagnostic | 解释 full SC-INR 有有限 analytic cell response，NoSinc 为 zero active response。 |
| 6.1-6.6 mechanism suite / SC-INR-EQ | `../diagnostics/mechanism_suite_2026-05-12/` | diagnostic/exploratory | 机制摘要；完整探索产物在 legacy archive。 |
| 用户确认 qualitative | `../paper_figures/qualitative_selected_seed1/` | candidate | 只作为 selected examples，不证明平均视觉质量。 |

## 不再作为活跃入口

- 旧 overview 输出：已被 canonical benchmark 入口替代。
- 未确认自动 qualitative 候选池：不能作为论文入口。
- 旧 evidence planning table：已被后续 evidence/benchmark 入口替代。

## 使用边界

辅助证据必须与 benchmark 主表分开解释。特别是 same-LR consistency 只能作为
diagnostic/path-invariance 证据；当 NoCell/NoSinc 负控获得更高 consistency 时，不能
把高 consistency 直接解释为 sampling correctness。
