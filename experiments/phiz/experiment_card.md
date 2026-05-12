# 实验：最终候选 SC-INR

## 科学问题

feature-conditioned phase 是否能改善 `SC-INR` 的重建质量，同时不重新引入 LTE 式
cell-conditioned phase extrapolation？

- 论文展示名：`SC-INR`
- canonical checkpoint：`artifacts/checkpoints/seed*/sc-inr`
- 历史 raw/checkpoint key：`SC-INR+PhiZ` / `sc-inr-phiz`

## 当前证据

- seed1 raw benchmark：`artifacts/raw_results/seed1/benchmark_signed_phiz.json`
- canonical seed1 context：`artifacts/derived/benchmarks/paper_context_seed1.csv`
- final 3-seed benchmark：`artifacts/derived/benchmarks/paper_main_3seed_summary.csv`
- 用户确认定性图：`artifacts/derived/paper_figures/qualitative_selected_seed1/`

## 当前结果

final 3-seed PSNR benchmark：

- `SC-INR - LIIF`：ID `+0.0437 dB`，OOD `+0.0328 dB`，ALL `+0.0364 dB`
- `SC-INR - LTE`：ID `-0.0048 dB`，OOD `+0.0504 dB`，ALL `+0.0320 dB`

seed1 结构增量背景：

- `SC-INR - LTE`：ID `+0.0738 dB`，OOD `+0.0849 dB`，ALL `+0.0812 dB`
- `SC-INR - SC-INR-NoPhi`：ID `+0.0673 dB`，OOD `+0.0295 dB`，ALL `+0.0421 dB`

用户确认 qualitative candidates：

- `urban100_img012_x8_delta_phiz_vs_lte.png`
- `urban100_img004_x8_delta_phiz_vs_lte.png`

## caveat

- `SC-INR` vs `SC-INR-NoPhi` 只有 seed1 context，不能写成 3 seed 主 claim。
- 最终 `SC-INR` 的 auxiliary consistency/texture metrics 仍主要是 seed1。
- selected qualitative figures 不能证明平均视觉质量更好。

## 下一步 gate

- 论文主表必须绑定 `artifacts/derived/benchmarks/`。
- 若要强化 sinc-response claim，需要先补 footprint correctness / mechanism diagnostics。
