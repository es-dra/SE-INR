# Benchmark Provenance

本目录保存已被 canonical benchmark 表吸收的 dated benchmark 阶段记录。

当前论文主 benchmark 入口是：

`artifacts/derived/benchmarks/`

本目录只用于审计 raw-key 迁移、阶段性统计和历史命令来源。不要从这里直接生成论文主表。

## 内容

- `benchmark_progress_2026-05-09/`：旧 no-phase/core seed1-3 阶段记录。
- `benchmark_progress_2026-05-10/`：NoSinc seed1 benchmark 消融 provenance。
- `benchmark_progress_2026-05-11/`：final `SC-INR` 3-seed 阶段记录。

## 使用边界

若本目录与 `artifacts/derived/benchmarks/` 数值解释冲突，以 canonical benchmark long table
和 `paper/claims_evidence_matrix.md` 为准。
