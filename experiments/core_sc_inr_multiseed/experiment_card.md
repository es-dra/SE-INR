# 实验：Core SC-INR-NoPhi 多 seed benchmark

## 科学问题

在相同 benchmark protocol 和多个 seed 下，no-phase `SC-INR-NoPhi` 是否相对 LTE
改善 OOD scale robustness？

## 冻结协议

- 模型：`LIIF`、`LTE`、`SC-INR-NoPhi`（旧 raw key `SC-INR`）。
- Seeds：1、2、3。
- 数据集：Set5、Set14、BSD100、Urban100。
- 尺度：x2、x3、x4、x6、x8、x12、x16、x24、x30。
- ID：x2/x3/x4。
- OOD：x6/x8/x12/x16/x24/x30。
- checkpoint 规则：`epoch-best.pth`。
- raw results：
  - `artifacts/raw_results/seed1/benchmark_signed_phiz.json`
  - `artifacts/raw_results/seed2/benchmark.json`
  - `artifacts/raw_results/seed3/benchmark.json`

## 结果

当前 derived summary：

- `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/multiseed_core_summary.csv`

`SC-INR-NoPhi` 相对 LTE：

- ID：`-0.0040 dB`
- OOD：`+0.0507 dB`
- ALL：`+0.0325 dB`

## claim 边界

该结果支持 no-phase 变体相对 LTE 有稳定小幅 OOD PSNR 改善。它不支持 strict
scale equivariance、大幅 SOTA claim，也不支持最终候选 `SC-INR` 的 multi-seed
superiority。
