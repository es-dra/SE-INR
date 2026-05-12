# Seed1 辅助指标与论文图表入口

全局结果阅读顺序建议先看 `artifacts/derived/evidence/README_zh.md` 和
`paper/ARTIFACTS_ALLOWED.md`，再进入本文档。本文档只解释
`artifacts/derived/diagnostics/seed1_aux_metrics_all8` 这一组辅助质量和跨尺度一致性结果。

这组结果用于支撑 `SC-INR-NoPhi` 的机制分析，不是最终多 seed 主 benchmark，也不是最终候选 `SC-INR` 的辅助指标。使用这些结果时，结论强度应限定为“seed1 上的辅助证据”。

## 这组结果回答什么

核心问题是：在保持重建质量基本不下降的前提下，`SC-INR-NoPhi` 是否能减少 LTE 类 decoder 对 output cell 的任意外推，并提升同一 LR 输入下的跨尺度观测一致性。

推荐论文表述：

> SC-INR-NoPhi 在 OOD scale 上相对 LTE 有小幅但稳定的质量提升，同时显著改善 same-LR cross-scale observation consistency。

不推荐表述：

> SC-INR 已经实现严格尺度等变。

## 数据与协议

- 数据集：BSD100、Urban100。
- 图像数：每个数据集前 10 张。
- 质量尺度：x4、x8、x16、x30。
- OOD 平均：x8、x16、x30 等权平均。
- Consistency 协议：同一个 LR 输入下，比较直接预测 x4 observation 与先预测 x8/x16/x30 再 bicubic downsample 到 x4 的结果。
- 基准参照：所有 delta 均在同数据集、同尺度或同 consistency pair 下相对 LTE 计算。

## 指标解释

- `PSNR-Y` / `SSIM-Y`：Y 通道重建质量。
- `texture RMSE`：GT 中 top 20% local variance 区域的 RMSE，越低越好。
- `highpass RMSE`：高通误差，越低越好；当前更适合作为补充诊断，不建议正文主打。
- `SC-PSNR`：same-LR cross-scale consistency PSNR，越高表示同一 LR 输入下不同输出采样路径更一致。
- `delta vs LTE`：相对 LTE 的差值。PSNR/SSIM/SC-PSNR 越高越好，RMSE 越低越好。

## 关键结论

- BSD100 OOD：SC-INR-NoPhi 相对 LTE 的 PSNR-Y 为 `+0.056 dB`，SSIM-Y 为 `+0.00191`。
- Urban100 OOD：SC-INR-NoPhi 相对 LTE 的 PSNR-Y 为 `+0.052 dB`，SSIM-Y 为 `+0.00101`。
- BSD100 consistency：SC-INR-NoPhi 相对 LTE 的 SC-PSNR 为 `+8.61 dB`，texture consistency RMSE 为 `-0.00590`。
- Urban100 consistency：SC-INR-NoPhi 相对 LTE 的 SC-PSNR 为 `+7.77 dB`，texture consistency RMSE 为 `-0.01047`。

重要 caveat：`LTE-NoCellPhase` 和 `LTE-PhaseZ` 的 consistency 很高，主要因为它们削弱了 cell response。它们是诊断性 baseline，不能仅凭 consistency 判定为更好的 ASISR 模型。

## 文件导航

- `quality_psnr_table.tex`：compact PSNR 表，可作为正文或补充表格基础。
- `quality_ssim_texture_table.tex`：OOD SSIM-Y 与 texture RMSE 表，更适合补充质量证据。
- `consistency_table.tex`：跨尺度观测一致性表，可作为机制分析主表。
- `quality_id_ood_with_lte_deltas.csv`：ID/OOD 聚合后的质量指标和 LTE delta。
- `quality_per_scale_with_lte_deltas.csv`：逐尺度质量指标和 LTE delta。
- `consistency_avg_with_lte_deltas.csv`：平均 consistency 指标和 LTE delta。
- `consistency_per_pair_with_lte_deltas.csv`：逐 source scale 的 consistency 指标和 LTE delta。
- `key_findings.md`：中文简要结论，便于快速恢复关键数字。

## 图表使用建议

适合正文改造：

- `fig_quality_consistency_tradeoff.pdf`：表达质量-一致性 tradeoff 的思路是对的，但需要重画。当前 NoCell/FeaturePhase 把 y 轴拉到 +25 dB，压缩了主模型差异；建议正文版本只保留主模型，诊断模型放 inset 或补充图。
- `fig_scale_gain_vs_lte.pdf`：可改成少模型曲线，只画 LTE、LTE-EQ、SC-INR-FixedOmega、SC-INR-NoPhi。

适合补充或调试，不建议直接进正文：

- `fig_ood_psnr_gain_vs_lte.pdf`
- `fig_ood_texture_rmse_delta_vs_lte.pdf`
- `../figures/quality_psnr_y.png`
- `../figures/texture_rmse_y.png`
- `../figures/highpass_rmse_y.png`
- `../figures/consistency_psnr_y.png`

原因：这些图更像实验监控图。柱状图过密、方法太多，读者难以抓住论文主结论。

## Qualitative 可视化重做要求

旧 `visual_crops/` debugging 输出已删除。后续应重做为配置化 qualitative figure：

- 左侧放完整 HR 图并用红框标出 crop。
- 右侧只放少量关键方法：`GT`、`Bicubic`、`LTE`、`LTE-EQ`、`SC-INR-FixedOmega`、`SC-INR-NoPhi`、最终候选 `SC-INR`。
- 正文图优先 x8 或 x12/x16；x30 放补充。
- 每个样例只放 1-2 个有结构差异的 crop，不自动选择最大 texture variance。
- error map 必须共享色域和 colorbar，不给 GT 放黑色误差块。
- 每个 crop 下可标局部 PSNR-Y / SSIM-Y 或 texture RMSE，但不要堆太多数字。

推荐后续输出：

- `main_qualitative_x8.pdf`：正文图，2 个样例，5-6 个方法。
- `supp_qualitative_all_methods.pdf`：补充图，包含 LIIF/EQ、NoCell、FeaturePhase 等诊断方法。

## 与其他结果的关系

- 主 benchmark 负责回答“重建质量有没有明显退化，以及 OOD PSNR 是否有小幅收益”。
- 连续尺度曲线负责回答“gain 是否随 scale 外推呈现更稳定的趋势”。
- 本组 auxiliary metrics 负责回答“same-LR output observation 是否更一致，以及纹理区域误差是否同步下降”。

因此，本组结果不应单独作为最终质量排名使用；它的价值在于支撑 sampling-consistent observation 机制。

## 下一步

1. 重画质量-一致性 tradeoff 图，区分主模型和诊断 baseline。
2. 写一个配置化 qualitative figure 脚本，人工指定 image/crop/methods。
3. seed2/seed3 完成后，用同样表格口径更新 mean/std。
4. 对最终候选 `SC-INR` 和 signed bounded omega/no-phase 变体补充 benchmark、response distribution 和 cell response curve。
