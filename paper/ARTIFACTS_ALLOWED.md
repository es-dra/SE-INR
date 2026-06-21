# 论文可引用产物白名单

本文档是当前 SC-INR 论文草稿的引用白名单，比 `MANIFEST.md` 更严格。
没有列在这里的产物，默认只能作为背景、诊断或候选材料，不能直接进入正文主证据。

## 当前活跃入口

| 用途 | Canonical 产物 | 状态 | caveat |
| --- | --- | --- | --- |
| 模型命名与配置 | `configs/registry/models.yaml`; `paper/model_taxonomy.md` | cite_ok | raw key、checkpoint 目录和论文展示名必须通过这里解释。 |
| 所有 benchmark long table | `artifacts/derived/benchmarks/benchmark_all_models_long.csv`; `artifacts/derived/benchmarks/README_zh.md` | cite_ok | 每行显式记录 `source_file/raw_key/canonical_model/seed/protocol_id`，避免直接读 raw key 误判。 |
| 论文主表 3-seed benchmark | `artifacts/derived/benchmarks/paper_main_3seed.csv`; `artifacts/derived/benchmarks/paper_main_3seed_summary.csv`; `artifacts/derived/benchmarks/paper_main_3seed_paired_delta.csv` | cite_ok | 只用于 LIIF/LTE/final `SC-INR` 主比较。 |
| seed1 context/diagnostic benchmark | `artifacts/derived/benchmarks/paper_context_seed1.csv` | context | 可展示 `SC-INR` vs `SC-INR-NoPhi` seed1 结构增量、NoSinc、EQ 和 LTE-side diagnostics；不能替代 3-seed 主表。 |
| 辅助证据总入口 | `artifacts/derived/evidence/README_zh.md` | cite_ok | 只作为索引；具体图表仍需引用下方 canonical artifact。 |

## Benchmark Provenance

以下 dated 目录保留为 provenance/support，不作为新的活跃入口扩散：

| 用途 | 产物 | 状态 | caveat |
| --- | --- | --- | --- |
| 旧 no-phase/core 3-seed benchmark | `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/multiseed_core_summary.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/multiseed_core_per_seed.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/README_zh.md` | support/audit | 主 benchmark 必须引用 `artifacts/derived/benchmarks/`；这里仅用于阶段审计。旧 raw JSON 中 `SC-INR` 表示 `SC-INR-NoPhi`。 |
| final-candidate SC-INR 3-seed provenance | `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-11/final_sc_inr_multiseed_summary.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-11/final_sc_inr_vs_liif_lte_paper.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-11/README_zh.md` | support/audit | 主 benchmark 必须引用 `artifacts/derived/benchmarks/`；这里保留原始阶段记录和 provenance。 |
| seed1 NoSinc benchmark provenance | `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-10/seed1_signed_phiz_nosinc_summary.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-10/seed1_nosinc_per_scale_delta.csv` | support | 单 seed 机制消融；不能写 sinc 唯一因果。 |

## 辅助证据

| 用途 | Canonical 产物 | 状态 | caveat |
| --- | --- | --- | --- |
| seed1 SC-INR-NoPhi 辅助一致性 | `artifacts/derived/diagnostics/seed1_aux_metrics_all8/paper/consistency_table.tex` | cite_ok | 旧 no-phase 主线；不包含最终候选 PhiZ。 |
| seed1 quality/texture 辅助表 | `artifacts/derived/diagnostics/seed1_aux_metrics_all8/paper/quality_ssim_texture_table.tex` | cite_ok | seed1 辅助协议。 |
| seed1 final-candidate SC-INR 辅助指标 | `artifacts/derived/diagnostics/sc_inr_final_aux_metrics_seed1/quality_summary.csv`; `artifacts/derived/diagnostics/sc_inr_final_aux_metrics_seed1/consistency_summary.csv`; `artifacts/derived/diagnostics/sc_inr_final_aux_metrics_seed1/README_zh.md` | preliminary | 覆盖 BSD100/Urban100 前 10 张；可写 preserved strong consistency vs LTE，但不能写 multi-seed 或 consistency 最强。 |
| seed1 SC-INR-NoSinc 辅助诊断 | `artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/quality_summary.csv`; `artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/consistency_summary.csv`; `artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/README_zh.md` | diagnostic | NoSinc same-LR self-consistency 显著更高，说明该指标不能单独证明 sinc response。 |
| response/omega 机制诊断 | `artifacts/derived/diagnostics/response_omega_diagnostics_2026-05-10/response_distribution_summary.csv`; `artifacts/derived/diagnostics/response_omega_diagnostics_2026-05-10/cell_response_curve_summary.csv`; `artifacts/derived/diagnostics/response_omega_diagnostics_2026-05-10/README_zh.md` | diagnostic | 小样本机制诊断；不能替代 benchmark 或多 seed。 |
| feature-level effective amplitude 诊断 | `artifacts/derived/diagnostics/effective_amplitude_2026-05-12/effective_amplitude_summary.csv`; `artifacts/derived/diagnostics/effective_amplitude_2026-05-12/README_zh.md` | diagnostic | 小样本机制诊断；只能写 decoder 输入层面 `q_coef(z) * W(omega(z), c)` 被 analytic response 调制，不能写最终 RGB 是严格 Fourier amplitude 或 exact box integral。 |
| footprint oracle 诊断 | `artifacts/derived/diagnostics/footprint_oracle_2026-05-15/footprint_oracle_overall.csv`; `artifacts/derived/diagnostics/footprint_oracle_2026-05-15/gate_conclusion.json`; `artifacts/derived/diagnostics/footprint_oracle_2026-05-15/README_zh.md` | diagnostic | seed1 小样本 HR box-average proxy；可写 `SC-INR` 比 `SC-INR-NoSinc` 更跟随 footprint oracle，不能写真实连续图像积分、sinc 唯一因果或 final variant 在所有辅助指标上最强。 |
| LTE vs SC-INR 统一机制诊断 | `artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/mechanism_overall.csv`; `artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/gate_conclusion.json`; `artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/README_zh.md` | diagnostic | seed1 crop-based 小样本机制 gate；可写 LTE 存在 learned `h_p(c)` cell-phase 信号、SC-INR 存在非零 analytic response/effective amplitude proxy，且在该 proxy oracle 上 tracking 略优于 LTE/NoSinc。不能替代全图 footprint oracle、benchmark 或 multi-seed 结论。 |
| cell intervention 可视化 | `artifacts/derived/diagnostics/cell_intervention_visual_2026-05-31/selected_examples.csv`; `artifacts/derived/diagnostics/cell_intervention_visual_2026-05-31/cell_intervention_metrics.csv`; `artifacts/derived/diagnostics/cell_intervention_visual_2026-05-31/figures/selected_01_urban100_img_008.png`; `artifacts/derived/diagnostics/cell_intervention_visual_2026-05-31/README_zh.md` | diagnostic/context | 周报友好的行为展示；固定同一 LR crop/query，只改变 cell multiplier，自动从 GT 高频 crop 中筛选候选。可写 selected examples 中 SC-INR 的 cell-induced change 更接近 HR box oracle 且 NoSinc 基本 cell-insensitive；不能写成全量 benchmark、平均视觉质量证明或 sinc 唯一因果。 |
| SC-INR 外部基线优势区间诊断 | `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/paired_delta_bootstrap_ci.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_per_crop.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_summary_by_scale.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_region_pool.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_x4_sanity_pool.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/README_zh.md`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/run_config.json` | diagnostic/context | 逐图统计是 first-10 中等预算诊断，不替代全量 3-seed benchmark；局部 crop 是 seed1 预注册候选池。正文主展示只用于 `SC-INR` vs `LIIF/LTE` 的 selected external-baseline advantage regions。`figures/external_advantage_crops/`、`figures/external_advantage_context_zoom/` 和 `figures/external_advantage_x4_sanity/` 是本地渲染候选图目录，默认不进源码管理；若某张图进入论文，应晋级到 `artifacts/derived/paper_figures/` 并单独登记。NoPhi/NoSinc 消融后续单独进入消融证据，不作为当前主图入口。x4 sanity 只能说明低倍率边界差异很小，不能写成 x4 明显优势。selected examples 必须与完整候选池统计一起解释，不能写成平均视觉质量证明。 |
| LIIF vs 普通 INR 机制图 | `artifacts/derived/paper_figures/liif_vs_inr_mechanism_2026-05-24/liif_vs_inr_mechanism.pdf`; `artifacts/derived/paper_figures/liif_vs_inr_mechanism_2026-05-24/README_zh.md` | candidate | 论文/周报机制图；只说明典型坐标 INR 与 LIIF 类局部特征条件化共享 decoder 的变量组织差异。不能写成普通 INR 不能超分、LIIF 不使用 cell，或 LIIF 具备严格尺度一致性。 |
| LTE vs SC-INR 机制对比图 | `artifacts/derived/paper_figures/lte_scinr_mechanism_2026-05-18/lte_scinr_mechanism_comparison.pdf`; `artifacts/derived/paper_figures/lte_scinr_mechanism_2026-05-18/README_zh.md` | candidate | 论文/周报机制图；只说明 cell 进入 decoder 的结构差异，应配合 `lte_scinr_mechanism_2026-05-16` 机制诊断使用。不能把图本身当作实验结果，也不能暗示 final RGB 是 exact box integral。 |
| SC-INR footprint response 机制图 | `artifacts/derived/paper_figures/footprint_response_mechanism_2026-05-31/footprint_response_mechanism.pdf`; `artifacts/derived/paper_figures/footprint_response_mechanism_2026-05-31/README_zh.md` | candidate | 论文/周报机制图；解释 `W(omega,c)` 的动机、sinc 来源和 cell 从 learned phase shortcut 到 analytic observation response 的语义转移。不能把概念图当作 benchmark、NoSinc 因果证据或最终 RGB exact box integral 证明。 |
| SC-INR-EQ seed1 benchmark 与 6.1-6.6 机制 suite | `artifacts/raw_results/seed1/benchmark_sc_inr_eq.json`; `artifacts/derived/diagnostics/mechanism_suite_2026-05-12/README_zh.md` | preliminary/diagnostic | exploratory Rot-E integration；完整探索产物在 legacy；可写相对 `LTE-EQ` 小幅正增益、相对 final `SC-INR` 基本打平，不能写成主方法或 multi-seed 结论。 |
| 用户确认的 qualitative 例图 | `artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img012_x8_selected_gt_bicubic_lte_nophi_scinr.png`; `artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img004_x8_selected_gt_bicubic_lte_nophi_scinr.png`; `artifacts/derived/paper_figures/qualitative_selected_seed1/README_zh.md` | candidate | 只能表述为 selected qualitative examples，不能写 average visual quality proof。 |

## 不允许作为正文主证据

- `artifacts/legacy/*`
- retired overview summaries
- unreviewed automatic qualitative candidate pools
- superseded evidence planning tables
- 未人工审查的 automatic top-delta crop 排名
- repo-local `Data/benchmark` paired-LR 结果，除非重新验证 LR/HR 一致性

命名映射以 `paper/model_taxonomy.md` 为准。特别注意：旧 raw key `SC-INR`
对应 `SC-INR-NoPhi`；最终候选 `SC-INR` 在 seed1 历史结果中对应 raw key
`SC-INR+PhiZ`，在 seed2/3 新评估文件中对应清理后的 raw key `SC-INR`。

## 晋级规则

新增产物进入白名单前，必须记录：

1. 来源 raw result 或 checkpoint；
2. 生成脚本或命令；
3. dataset、scale、seed 和 protocol；
4. 已知 caveat；
5. 允许使用的论文表述。
