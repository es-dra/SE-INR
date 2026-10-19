# 论文产物范围

当前分析论文的来源主本是[current/SOURCES.md](current/SOURCES.md)，实验文本及完整逐条件评分见[October证据](../experiments/october2026/)，[manifest](../experiments/october2026/manifest.json)记录逐文件及原件包SHA。正结果、失败假设、反例和开发/正式划分必须共同保留；核验范围按各独立审查原件说明，不能概称所有模型全部重放。

下方为旧SC-INR新方法内部记录白名单，原文保留。其“不能恢复投稿claim”针对新方法身份；当前失败诊断分析稿不据此宣称新模型优势。checkpoint/数据/原始日志及大型数组保持服务器原位，不批量提交公开仓库。

---

# SC-INR 内部技术记录产物白名单

本文档约束冻结 SC-INR 技术记录可以引用的产物，比 `MANIFEST.md` 更严格。P1.12 双 baseline
硬门失败后，本表中的 `cite_ok` 统一解释为 `internal-report-ok`：只允许支撑内部 closeout、复现说明
和历史技术记录，不允许恢复新方法或投稿 claim。没有列在这里的产物，默认只能作为背景、诊断或
候选材料，不能进入核心证据。

## 冻结证据入口

| 用途 | Canonical 产物 | 状态 | caveat |
| --- | --- | --- | --- |
| 模型命名与配置 | `configs/registry/models.yaml`; `paper/model_taxonomy.md` | cite_ok | raw key、checkpoint 目录和论文展示名必须通过这里解释。 |
| 所有 benchmark long table | `artifacts/derived/benchmarks/benchmark_all_models_long.csv`; `artifacts/derived/benchmarks/README_zh.md` | cite_ok | 每行显式记录 `source_file/raw_key/canonical_model/seed/protocol_id`，避免直接读 raw key 误判。 |
| 三 seed benchmark | `artifacts/derived/benchmarks/paper_main_3seed.csv`; `artifacts/derived/benchmarks/paper_main_3seed_summary.csv`; `artifacts/derived/benchmarks/paper_main_3seed_paired_delta.csv` | cite_ok | 只用于记录 LIIF/LTE/冻结 `SC-INR` 在当前协议下的实现差异。 |
| seed1 context/diagnostic benchmark | `artifacts/derived/benchmarks/paper_context_seed1.csv` | context | 可展示 `SC-INR` vs `SC-INR-NoPhi` seed1 结构增量、NoSinc、EQ 和 LTE-side diagnostics；不能替代 3-seed 主表。 |
| 辅助证据总入口 | `artifacts/derived/evidence/README_zh.md` | cite_ok | 只作为索引；具体图表仍需引用下方 canonical artifact。 |

## Benchmark Provenance

以下 dated 目录保留为 provenance/support，不作为新的活跃入口扩散：

| 用途 | 产物 | 状态 | caveat |
| --- | --- | --- | --- |
| 旧 no-phase/core 3-seed benchmark | `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/multiseed_core_summary.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/multiseed_core_per_seed.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/README_zh.md` | support/audit | 主 benchmark 必须引用 `artifacts/derived/benchmarks/`；这里仅用于阶段审计。旧 raw JSON 中 `SC-INR` 表示 `SC-INR-NoPhi`。 |
| 历史 SC-INR 3-seed provenance | `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-11/final_sc_inr_multiseed_summary.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-11/final_sc_inr_vs_liif_lte_paper.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-11/README_zh.md` | support/audit | canonical benchmark 必须引用 `artifacts/derived/benchmarks/`；这里保留原始阶段记录和 provenance。 |
| seed1 NoSinc benchmark provenance | `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-10/seed1_signed_phiz_nosinc_summary.csv`; `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-10/seed1_nosinc_per_scale_delta.csv` | support | 单 seed 机制消融；不能写 sinc 唯一因果。 |

## 辅助证据

| 用途 | Canonical 产物 | 状态 | caveat |
| --- | --- | --- | --- |
| seed1 SC-INR-NoPhi 辅助一致性 | `artifacts/derived/diagnostics/seed1_aux_metrics_all8/paper/consistency_table.tex` | cite_ok | 旧 no-phase 主线；不包含最终候选 PhiZ。 |
| seed1 quality/texture 辅助表 | `artifacts/derived/diagnostics/seed1_aux_metrics_all8/paper/quality_ssim_texture_table.tex` | cite_ok | seed1 辅助协议。 |
| INCODE 启发的 cell-phase proxy | `artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_phase_proxy/cell_phase_proxy.csv`; `artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_phase_proxy/README_zh.md` | diagnostic | 无训练机制诊断；可写 LTE 有显式 `cell -> phase` 路径，SC-INR/LTE-PhaseZ/NoSinc 没有。不能证明 LTE 一定产生视觉错误。 |
| INCODE 启发的 effective amplitude / response | `artifacts/derived/diagnostics/incode_method_support_2026-06-20/effective_amplitude/effective_amplitude_summary.csv`; `artifacts/derived/diagnostics/incode_method_support_2026-06-20/effective_amplitude/README_zh.md` | diagnostic | 小样本 feature-level proxy；只能写 decoder 输入层 `q_coef(z) * W(omega(z), c)` 被 footprint response 调制，不能写最终 RGB 严格频谱振幅或 exact box integral。 |
| INCODE 启发的 cell-only intervention | `artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_intervention/selected_examples.csv`; `artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_intervention/cell_intervention_metrics.csv`; `artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_intervention/README_zh.md` | diagnostic/context | 固定 x4 LR/query 只改变 cell；可写 selected examples 中 SC-INR 有非零 cell response 且区别于 LTE/NoSinc。小样本均值没有稳定超过 LIIF，不能写平均视觉质量证明。 |
| SC-INR 外部基线优势区间诊断 | `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/paired_delta_bootstrap_ci.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_per_crop.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_summary_by_scale.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_region_pool.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_x4_sanity_pool.csv`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/README_zh.md`; `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/run_config.json` | diagnostic/context | 逐图统计是 first-10 中等预算诊断，不替代全量 3-seed benchmark；局部 crop 是 seed1 预注册候选池。正文主展示只用于 `SC-INR` vs `LIIF/LTE` 的 selected external-baseline advantage regions。`figures/external_advantage_crops/`、`figures/external_advantage_context_zoom/` 和 `figures/external_advantage_x4_sanity/` 是本地渲染候选图目录，默认不进源码管理；若某张图进入论文，应晋级到 `artifacts/derived/paper_figures/` 并单独登记。NoPhi/NoSinc 消融后续单独进入消融证据，不作为当前主图入口。x4 sanity 只能说明低倍率边界差异很小，不能写成 x4 明显优势。selected examples 必须与完整候选池统计一起解释，不能写成平均视觉质量证明。 |
| LIIF vs 普通 INR 机制图 | `artifacts/derived/paper_figures/liif_vs_inr_mechanism_2026-05-24/README_zh.md` | local_regenerate_required | 仅保留图说明；PDF/PNG/SVG 是本地可重渲染输出，进入论文前需要重渲染、人工检查并晋级为具体 artifact。不能写成普通 INR 不能超分、LIIF 不使用 cell，或 LIIF 具备严格尺度一致性。 |
| LTE vs SC-INR 机制对比图 | `artifacts/derived/paper_figures/lte_scinr_mechanism_2026-05-18/lte_scinr_mechanism_comparison.pdf`; `artifacts/derived/paper_figures/lte_scinr_mechanism_2026-05-18/README_zh.md` | candidate | 论文/周报机制图；只说明 cell 进入 decoder 的结构差异。不能把图本身当作实验结果，也不能暗示 final RGB 是 exact box integral。 |
| SC-INR-EQ seed1 benchmark | `artifacts/raw_results/seed1/benchmark_sc_inr_eq.json`; `artifacts/derived/benchmarks/paper_context_seed1.csv` | preliminary/context | exploratory Rot-E integration；可写相对 `LTE-EQ` 小幅正增益、相对 final `SC-INR` 基本打平，不能写成主方法或 multi-seed 结论。 |
| 用户确认的 qualitative 例图 | `artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img012_x8_selected_gt_bicubic_lte_nophi_scinr.png`; `artifacts/derived/paper_figures/qualitative_selected_seed1/urban100_img004_x8_selected_gt_bicubic_lte_nophi_scinr.png`; `artifacts/derived/paper_figures/qualitative_selected_seed1/README_zh.md` | candidate | 只能表述为 selected qualitative examples，不能写 average visual quality proof。 |
| P1.7 deterministic detail no-go | `artifacts/derived/deterministic_detail_no_go_p17/`; `docs/project/deterministic_detail_p17_theory_audit_2026-08-06.md` | research_gate/not_main_paper | 仅支持 frozen finite-spline alias pair 的 identifiability boundary；不能外推自然图像、概率模型或 learned architecture，也不能作为当前 SC-INR 主论文的方法贡献。 |
| P1.8 conditional-mean fiber theorem | `artifacts/derived/conditional_mean_intertwining_p18/`; `docs/project/conditional_mean_intertwining_p18_theory_audit_2026-08-06.md` | research_gate/not_main_paper | 仅支持 finite-support theorem fixture、joint-vs-marginal 反例和 P1.6/P1.7 applicability boundary；不能写成自然图像 joint closure、physical conditional mean 已被反证、概率模型已验证或 learned architecture contribution。 |
| P1.10 multi-observation observability | `artifacts/derived/multi_observation_observability_p110/`; `docs/project/multi_observation_observability_p110_theory_audit_2026-08-06.md` | research_gate/not_main_paper | 仅支持冻结 spline/box-sensor 与预注册 bundle 的 target-aware negative gate；不能外推多帧 SR、其他 acquisition、自然图像恢复或 learned INR，也不是新 generalized-sampling theorem。 |

## 需重跑后才能重新引用

以下旧路径当前不在文件树中，不能继续作为当前论文证据。若后续需要其中结论，必须重跑脚本、
检查产物、更新本白名单和 claim matrix：

| 历史用途 | 缺失路径 | 状态 |
| --- | --- | --- |
| final SC-INR seed1 auxiliary | `artifacts/derived/diagnostics/sc_inr_final_aux_metrics_seed1/` | missing/rerun_required |
| NoSinc auxiliary caveat | `artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/` | missing/rerun_required |
| response/omega 机制诊断 | `artifacts/derived/diagnostics/response_omega_diagnostics_2026-05-10/` | missing/rerun_required |
| 旧 effective amplitude 诊断 | `artifacts/derived/diagnostics/effective_amplitude_2026-05-12/` | missing/rerun_required；当前替代入口是 `incode_method_support_2026-06-20/effective_amplitude/` |
| footprint oracle 诊断 | `artifacts/derived/diagnostics/footprint_oracle_2026-05-15/` | missing/rerun_required |
| LTE vs SC-INR 统一机制诊断 | `artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/` | missing/rerun_required |
| cell intervention 旧可视化 | `artifacts/derived/diagnostics/cell_intervention_visual_2026-05-31/` | missing/rerun_required；当前替代入口是 `incode_method_support_2026-06-20/cell_intervention/` |
| SC-INR-EQ mechanism suite active mirror | `artifacts/derived/diagnostics/mechanism_suite_2026-05-12/` | missing/rerun_required；完整历史产物在 `artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/` |
| footprint response 机制图 | `artifacts/derived/paper_figures/footprint_response_mechanism_2026-05-31/` | missing/rerun_required |

## 不允许作为正文主证据

- `artifacts/legacy/*`
- retired overview summaries
- unreviewed automatic qualitative candidate pools
- superseded evidence planning tables
- 未人工审查的 automatic top-delta crop 排名
- repo-local `Data/benchmark` paired-LR 结果，除非重新验证 LR/HR 一致性
- 上方列为 `missing/rerun_required` 的历史路径

命名映射以 `paper/model_taxonomy.md` 为准。特别注意：旧 raw key `SC-INR`
对应 `SC-INR-NoPhi`；冻结的 canonical `SC-INR` 在 seed1 历史结果中对应 raw key
`SC-INR+PhiZ`，在 seed2/3 新评估文件中对应清理后的 raw key `SC-INR`。

## 恢复与晋级规则

当前没有 artifact 晋级为投稿证据的自动路径。外部 provenance 改变时，必须先重新通过 P1.12 并
建立新的 Node Contract；随后仍需直接 baseline、公平协议、factorial attribution 和有效 observation
正负控，才能重新讨论窄 claim。

新增产物进入白名单前，必须记录：

1. 来源 raw result 或 checkpoint；
2. 生成脚本或命令；
3. dataset、scale、seed 和 protocol；
4. 已知 caveat；
5. 允许使用的论文表述。
