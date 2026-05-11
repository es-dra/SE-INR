# 论文可引用产物白名单

本文档是当前 SC-INR 论文草稿的引用白名单，比 `MANIFEST.md` 更严格。
没有列在这里的产物，默认只能作为背景、诊断或候选材料，不能直接进入正文主证据。

## 当前表图来源

| 用途 | Canonical 产物 | 状态 | caveat |
| --- | --- | --- | --- |
| 核心 3-seed benchmark | `artifacts/derived/analysis/benchmark_progress_2026-05-09/multiseed_core_summary.csv` | cite_ok | 只覆盖 LIIF/LTE/SC-INR-NoPhi；旧 raw JSON 中 `SC-INR` 表示 `SC-INR-NoPhi`。 |
| 核心 benchmark per-seed 明细 | `artifacts/derived/analysis/benchmark_progress_2026-05-09/multiseed_core_per_seed.csv` | cite_ok | 应和 mean/std 汇总一起使用。 |
| final-candidate SC-INR 3-seed benchmark | `artifacts/derived/analysis/benchmark_progress_2026-05-11/final_sc_inr_multiseed_summary.csv`; `artifacts/derived/analysis/benchmark_progress_2026-05-11/final_sc_inr_vs_liif_lte_paper.csv`; `artifacts/derived/analysis/benchmark_progress_2026-05-11/README_zh.md` | cite_ok | seed1 raw key 为 `SC-INR+PhiZ`，seed2/3 清理后 raw key 为 `SC-INR`；论文主表优先展示相对 LIIF/LTE 的 3-seed 差距。 |
| seed1 final-candidate SC-INR vs NoPhi benchmark | `artifacts/derived/analysis/benchmark_progress_2026-05-09/seed1_signed_phiz_summary.csv` | context | raw key 为 `SC-INR+PhiZ`；只用于展示 seed1 上相对 `SC-INR-NoPhi` 的结构增量，不替代 3-seed LIIF/LTE 主表。 |
| seed1 SC-INR-NoPhi 辅助一致性 | `artifacts/derived/analysis/seed1_aux_metrics_all8/paper/consistency_table.tex` | cite_ok | 旧 no-phase 主线；不包含最终候选 PhiZ。 |
| seed1 quality/texture 辅助表 | `artifacts/derived/analysis/seed1_aux_metrics_all8/paper/quality_ssim_texture_table.tex` | cite_ok | seed1 辅助协议。 |
| seed1 final-candidate SC-INR 辅助指标 | `artifacts/derived/analysis/sc_inr_final_aux_metrics_seed1/quality_summary.csv`; `artifacts/derived/analysis/sc_inr_final_aux_metrics_seed1/consistency_summary.csv`; `artifacts/derived/analysis/sc_inr_final_aux_metrics_seed1/README_zh.md` | preliminary | 覆盖 BSD100/Urban100 前 10 张；可写 preserved strong consistency vs LTE，但不能写 multi-seed 或 consistency 最强。 |
| seed1 SC-INR-NoSinc benchmark | `artifacts/derived/analysis/benchmark_progress_2026-05-10/seed1_signed_phiz_nosinc_summary.csv`; `artifacts/derived/analysis/benchmark_progress_2026-05-10/seed1_nosinc_per_scale_delta.csv` | preliminary | 单 seed 机制消融；可写 full benchmark 中 NoSinc 弱于完整 SC-INR，不能写 sinc 唯一因果。 |
| seed1 SC-INR-NoSinc 辅助诊断 | `artifacts/derived/analysis/sc_inr_nosinc_aux_metrics_seed1/quality_summary.csv`; `artifacts/derived/analysis/sc_inr_nosinc_aux_metrics_seed1/consistency_summary.csv`; `artifacts/derived/analysis/sc_inr_nosinc_aux_metrics_seed1/README_zh.md` | diagnostic | NoSinc same-LR self-consistency 显著更高，说明该指标不能单独证明 sinc response；应作为 caveat 使用。 |
| response/omega 机制诊断 | `artifacts/derived/analysis/response_omega_diagnostics_2026-05-10/response_distribution_summary.csv`; `artifacts/derived/analysis/response_omega_diagnostics_2026-05-10/cell_response_curve_summary.csv`; `artifacts/derived/analysis/response_omega_diagnostics_2026-05-10/README_zh.md` | diagnostic | 小样本机制诊断；可解释 NoSinc 高 self-consistency 来自 zero active cell response，不能替代 benchmark 或多 seed。 |
| 用户确认的 qualitative 例图 | `artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img012_x8_selected_gt_bicubic_lte_nophi_scinr.png`; `artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img004_x8_selected_gt_bicubic_lte_nophi_scinr.png`; `artifacts/derived/paper_figures/qualitative_selected_seed1/README_zh.md` | candidate | 已用当前命名重新导出；只能表述为 selected qualitative examples，不能写 average visual quality proof。 |
| 当前 evidence table planning | `artifacts/derived/paper_tables/current_evidence_2026-05-11/current_evidence_table.csv`; `artifacts/derived/paper_tables/current_evidence_2026-05-11/README_zh.md` | planning | 只汇总已有 canonical artifacts，不引入新 benchmark；用于 LaTeX 表格规划，不替代源产物。 |

命名映射以 `paper/model_taxonomy.md` 为准。特别注意：旧 raw key `SC-INR`
对应 `SC-INR-NoPhi`；最终候选 `SC-INR` 在 seed1 历史结果中对应 raw key
`SC-INR+PhiZ`，在 seed2/3 新评估文件中对应清理后的 raw key `SC-INR`。

## 不允许作为正文主证据

- `artifacts/smoke/*`
- `artifacts/legacy/*`
- `artifacts/derived/analysis/overview/benchmark_seed_mean_std.csv` 作为“最终候选 SC-INR 的 3-seed 证明”
- 未人工审查的 automatic top-delta crop 排名
- repo-local `Data/benchmark` paired-LR 结果，除非重新验证 LR/HR 一致性

## 晋级规则

新增产物进入白名单前，必须记录：

1. 来源 raw result 或 checkpoint；
2. 生成脚本或命令；
3. dataset、scale、seed 和 protocol；
4. 已知 caveat；
5. 允许使用的论文表述。
