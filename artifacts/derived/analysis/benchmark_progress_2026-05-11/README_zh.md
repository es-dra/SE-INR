# 2026-05-11 final-candidate SC-INR 多 seed benchmark 汇总

本目录只汇总最终候选 `SC-INR` 的 3 seed benchmark，不覆盖旧的 raw JSON。

## 原始结果来源

- seed1: `artifacts/raw_results/seed1/benchmark_signed_phiz.json`，raw key 为历史兼容名 `SC-INR+PhiZ`。
- seed2: `artifacts/raw_results/seed2/benchmark_sc_inr.json`，raw key 为清理后的 `SC-INR`。
- seed3: `artifacts/raw_results/seed3/benchmark_sc_inr.json`，raw key 为清理后的 `SC-INR`。

三个 seed 均覆盖 `Set5/Set14/BSD100/Urban100` × `x2/x3/x4/x6/x8/x12/x16/x24/x30`，共 36 项，未发现缺失或 `None`。

## 派生产物

- `final_sc_inr_multiseed_per_seed.csv`: 每个 seed 的 ID/OOD/ALL PSNR。
- `final_sc_inr_multiseed_summary.csv`: 3 seed mean/std/min/max。
- `final_sc_inr_scale_summary.csv`: 每个 scale 的 3 seed mean/std。
- `final_sc_inr_vs_liif_lte_paper.csv`: 论文展示优先使用的 `SC-INR` vs `LIIF`/`LTE` paired seed delta。
- `final_sc_inr_vs_core_paired_delta.csv`: 内部审计用，包含与 `LIIF`、`LTE`、旧 `SC-INR-NoPhi` 的 paired seed delta。
- `provenance.json`: raw 文件、checkpoint、命令、覆盖范围和 caveat。

## 关键数值

- `SC-INR` final 3 seed: ID `31.0738 ± 0.0582`，OOD `22.7578 ± 0.0316`，ALL `25.5298 ± 0.0403`。
- 相对 `LIIF` paired delta: ID `+0.0437 ± 0.0757`，OOD `+0.0328 ± 0.0373`，ALL `+0.0364 ± 0.0499`。
- 相对 `LTE` paired delta: ID `-0.0048 ± 0.0836`，OOD `+0.0504 ± 0.0417`，ALL `+0.0320 ± 0.0555`。

## 当前可支持的结论

最终候选 `SC-INR` 的 3 seed PSNR benchmark 已补齐。论文展示优先比较 `SC-INR` 与 `LIIF`、`LTE`：它相对 `LIIF` 在 ID/OOD/ALL 上均为正均值，相对 `LTE` 在 OOD 和 ALL 上为正均值、ID 基本持平。`SC-INR` 与旧 `SC-INR-NoPhi` 的 seed2/seed3 差异只作为内部审计，不建议放入主表；若需要说明 feature-conditioned phase 的结构增量，用 seed1 结果即可。

## 仍需限制的表述

这些表只证明 benchmark PSNR 层面的 3 seed 复现，不证明 strict scale equivariance，也不替代 consistency、texture RMSE、continuous-scale curve 和机制诊断。论文主张应仍绑定到 decoder-side sampling consistency 与 OOD robustness，避免把 feature phase 写成已完全解释的理论贡献。
