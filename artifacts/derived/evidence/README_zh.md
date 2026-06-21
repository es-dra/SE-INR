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
| effective amplitude diagnostics | `../diagnostics/effective_amplitude_2026-05-12/` | diagnostic | 直接统计 feature-level `q_coef(z) * W(omega(z), c)`，说明 cell 作用在有效输入强度而非 phase。 |
| footprint oracle diagnostics | `../diagnostics/footprint_oracle_2026-05-15/` | diagnostic | seed1 小样本 proxy；固定 x4 LR/query，只改变 cell multiplier，并与 HR box-average proxy 比较；支持区分 analytic response 与 cell-insensitive 负控，但不是 exact integral 证明。 |
| LTE vs SC-INR 统一机制诊断 | `../diagnostics/lte_scinr_mechanism_2026-05-16/` | diagnostic | seed1 crop-based proxy；在同一 crop/query/cell 协议下并列检查 LTE `h_p(c)`、SC-INR active response/effective amplitude proxy 和 footprint oracle；支持主脉络机制解释，但不替代 benchmark。 |
| cell intervention 可视化 | `../diagnostics/cell_intervention_visual_2026-05-31/` | diagnostic/context | 周报友好的行为展示：固定同一 LR crop 和 query，只把 cell 从 native 放大到 x4；自动从 GT 高频 crop 中筛选候选，展示 LTE、SC-INR、NoSinc 对 cell 的不同响应。用于说明 SC-INR 更接近 footprint oracle 的变化，不能替代 benchmark 或证明全局视觉质量更好。 |
| SC-INR 外部基线优势区间诊断 | `../diagnostics/sc_inr_advantage_midbudget10_2026-05-21/` | diagnostic/context | first-10 逐图 paired statistics + seed1 预注册局部 crop 候选池；主用途是说明 `SC-INR` 相对 `LIIF/LTE` 的 OOD win-rate/median delta 和 selected external-baseline advantage regions。context+zoom 图池默认本地保留，论文采用前需迁移到 `../paper_figures/` 并登记；x4 sanity 只作为低倍率边界对照。不替代全量主 benchmark；NoPhi/NoSinc 相关 CSV 仅作内部诊断或后续消融。 |
| LTE vs SC-INR 机制对比图 | `../paper_figures/lte_scinr_mechanism_2026-05-18/` | candidate | 论文/周报用结构示意图；用于解释形式差异，不能替代机制诊断或 benchmark。 |
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
footprint oracle 是 HR 离散图上的 box-average proxy，也不能写成真实连续图像积分证明。
