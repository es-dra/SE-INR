# 2026-05-10 NoSinc seed1 benchmark 消融

本目录是 `SC-INR-NoSinc` 的 seed1 benchmark 阶段记录。它保留为机制消融证据，
不作为论文主 benchmark 入口；当前主 benchmark 入口见
`artifacts/derived/benchmarks/`。

## 原始结果来源

- `artifacts/raw_results/seed1/benchmark_signed_phiz.json`
  - 包含 LIIF/LTE、LTE 诊断模型、SC-INR-NoPhi、SC-INR-NoPhi-Signed 和最终候选 `SC-INR`。
  - 历史 raw key `SC-INR` 表示 `SC-INR-NoPhi`，`SC-INR+PhiZ` 表示最终候选。
- `artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json`
  - 包含 `SC-INR-NoSinc`。

## 派生产物

- `seed1_signed_phiz_nosinc_summary.csv`：seed1 各模型 ID/OOD/ALL PSNR 及相对 LTE、最终 `SC-INR`、`SC-INR-NoPhi` 的差值。
- `seed1_nosinc_per_scale_delta.csv`：`SC-INR-NoSinc` 和相关 SC-INR 变体按 dataset/scale 的差值。
- `seed1_signed_phiz_nosinc_provenance.json`：输入文件、评估协议和命名 caveat。

## 使用边界

可支持的谨慎表述：在 seed1 benchmark 上，去掉 analytic sinc response 后相对最终
`SC-INR` 的 ID/OOD/ALL PSNR 分别下降 `0.1397/0.0467/0.0777 dB`。

不可支持的表述：不能据此宣称 sinc 是唯一因果因素，也不能把 NoSinc 的 same-LR
consistency 结果解释为机制失败。NoSinc 的 consistency caveat 见
`artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/`。
