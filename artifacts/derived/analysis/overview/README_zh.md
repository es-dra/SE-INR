# 结果分析入口

本文档是兼容路径 `results/analysis/README_zh.md` 的中文入口。真实结果已按
论文级研究工程结构迁移到 `artifacts/`，旧 `results/...` 路径通过 symlink
保留。

## 当前主入口

- 项目总览：`README.md`
- 全局产物清单：`MANIFEST.md`
- artifact 分层说明：`artifacts/README_zh.md`
- 模型别名：`configs/registry/models.yaml`
- 协议清单：`configs/registry/protocols.yaml`
- claim-evidence ledger：`paper/claims_evidence_matrix.md`

## 当前可信证据

| 证据 | Canonical 路径 | 兼容路径 | 状态 |
| --- | --- | --- | --- |
| seed1 benchmark | `artifacts/raw_results/seed1/benchmark.json` | `results/benchmark.json` | formal raw |
| seed1 Signed/PhiZ benchmark | `artifacts/raw_results/seed1/benchmark_signed_phiz.json` | `results/benchmark_seed1_with_signed_phiz.json` | formal raw |
| seed2 core benchmark | `artifacts/raw_results/seed2/benchmark.json` | `results/seeds/benchmark_seed2.json` | formal raw |
| seed3 core benchmark | `artifacts/raw_results/seed3/benchmark.json` | `results/seeds/benchmark_seed3.json` | formal raw |
| 当前 overview benchmark 汇总 | `artifacts/derived/analysis/overview/benchmark_summary.csv` | `results/analysis/overview/benchmark_summary.csv` | derived |
| 当前 overview 多 seed 汇总 | `artifacts/derived/analysis/overview/benchmark_seed_mean_std.csv` | `results/analysis/overview/benchmark_seed_mean_std.csv` | derived |
| 2026-05-09 阶段性人工汇总 | `artifacts/derived/analysis/benchmark_progress_2026-05-09/` | `results/analysis/benchmark_progress_2026-05-09/` | derived |
| seed1 auxiliary metrics | `artifacts/derived/analysis/seed1_aux_metrics_all8/` | `results/analysis/seed1_aux_metrics_all8/` | derived |
| PhiZ qualitative candidates | `artifacts/derived/paper_candidates/qualitative_phiz_candidates/` | `results/analysis/qualitative_phiz_candidates/` | candidate |

## 关键结论边界

- 核心 `SC-INR-NoPhi`（raw key `SC-INR`）的 3 seed OOD PSNR 相对 `LTE` 有稳定小幅提升。
- 最终候选 `SC-INR`（raw key `SC-INR+PhiZ`）在 seed1 PSNR 上是强候选，并有两张用户确认的局部视觉改善图。
- seed1 auxiliary metrics 支持 decoder-side same-LR cross-scale observation consistency。
- 当前仍不能宣称 strict scale equivariance。
- 最终候选 `SC-INR` 还需要 auxiliary metrics、多 seed 和 w/o-sinc 消融，才能作为最终主模型。

## 不应正式引用

- `artifacts/smoke/*`
- `artifacts/legacy/*`
- 自动 top-delta qualitative crop 作为平均视觉质量证明

注意：`overview/benchmark_seed_mean_std.csv` 中最终候选 `SC-INR` 仍只有 seed1；
3 seed 统计目前只适合支持 `LIIF`、`LTE`、`SC-INR-NoPhi` 的核心稳定性比较。

## 下一步

1. 为最终候选 `SC-INR` 跑 auxiliary metrics。
2. 设计 `SC-INR w/o sinc` 消融。
3. 用 `paper/claims_evidence_matrix.md` 约束论文表述。
