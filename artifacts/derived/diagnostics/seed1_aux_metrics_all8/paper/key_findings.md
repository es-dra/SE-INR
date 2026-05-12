# Seed1 辅助指标要点

输入来自 `seed1_aux_metrics_all8` 的 `quality_summary.csv` 和
`consistency_summary.csv`。质量 OOD 均值使用 x8、x16、x30；consistency 均值使用
x8/x16/x30 到 x4 的 same-LR 路径。

## SC-INR-NoPhi 相对 LTE

- BSD100 x4 PSNR：`+0.004 dB`。
- BSD100 OOD PSNR：`+0.056 dB`；OOD SSIM：`+0.00191`。
- BSD100 OOD texture RMSE：`-0.00095`。
- BSD100 consistency PSNR：`+8.61 dB`；texture consistency RMSE：`-0.00590`。
- Urban100 x4 PSNR：`+0.031 dB`。
- Urban100 OOD PSNR：`+0.052 dB`；OOD SSIM：`+0.00101`。
- Urban100 OOD texture RMSE：`-0.00094`。
- Urban100 consistency PSNR：`+7.77 dB`；texture consistency RMSE：`-0.01047`。

## 解释边界

`LTE-NoCellPhase` 和 `LTE-PhaseZ` 的 cross-scale consistency 更高，主要因为它们对
cell size 的响应更弱。除非同时看 reconstruction quality 和 texture error，不能把
高 consistency 直接解释为更好的 ASISR 模型。

## 论文用途

- `quality_psnr_table.tex`：紧凑 PSNR 表。
- `quality_ssim_texture_table.tex`：SSIM 和 texture 支撑。
- `consistency_table.tex`：same-LR cross-scale observation consistency。
- `fig_quality_consistency_tradeoff.pdf`：展示 quality / consistency tradeoff。
