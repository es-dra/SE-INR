# 论文可引用产物白名单

本文档是当前 SC-INR 论文草稿的引用白名单，比 `MANIFEST.md` 更严格。
没有列在这里的产物，默认只能作为背景、诊断或候选材料，不能直接进入正文主证据。

## 当前表图来源

| 用途 | Canonical 产物 | 状态 | caveat |
| --- | --- | --- | --- |
| 核心 3-seed benchmark | `artifacts/derived/analysis/benchmark_progress_2026-05-09/multiseed_core_summary.csv` | cite_ok | 只覆盖核心 LIIF/LTE/SC-INR。 |
| 核心 benchmark per-seed 明细 | `artifacts/derived/analysis/benchmark_progress_2026-05-09/multiseed_core_per_seed.csv` | cite_ok | 应和 mean/std 汇总一起使用。 |
| seed1 Signed/PhiZ benchmark | `artifacts/derived/analysis/benchmark_progress_2026-05-09/seed1_signed_phiz_summary.csv` | preliminary | 新变体目前只有单 seed。 |
| seed1 SC-INR 辅助一致性 | `artifacts/derived/analysis/seed1_aux_metrics_all8/paper/consistency_table.tex` | cite_ok | 暂不包含 PhiZ。 |
| seed1 quality/texture 辅助表 | `artifacts/derived/analysis/seed1_aux_metrics_all8/paper/quality_ssim_texture_table.tex` | cite_ok | seed1 辅助协议。 |
| 用户确认的 qualitative 例图 | `artifacts/derived/paper_candidates/qualitative_phiz_candidates/urban100_img012_x8_delta_phiz_vs_lte.png`; `artifacts/derived/paper_candidates/qualitative_phiz_candidates/urban100_img004_x8_delta_phiz_vs_lte.png` | candidate | 只能表述为 selected examples。 |

## 不允许作为正文主证据

- `artifacts/smoke/*`
- `artifacts/legacy/*`
- `artifacts/derived/analysis/overview/benchmark_seed_mean_std.csv`
- 未人工审查的 automatic top-delta crop 排名
- repo-local `Data/benchmark` paired-LR 结果，除非重新验证 LR/HR 一致性

## 晋级规则

新增产物进入白名单前，必须记录：

1. 来源 raw result 或 checkpoint；
2. 生成脚本或命令；
3. dataset、scale、seed 和 protocol；
4. 已知 caveat；
5. 允许使用的论文表述。
