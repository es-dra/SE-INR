# 结果分析入口

本文档是当前 `results/analysis/` 的中文阅读入口，目标是说明哪些结果可以支撑论文叙事、哪些只是诊断材料，以及下一阶段应如何补强。当前结论以单 seed 和已有辅助分析为主，不能替代最终多 seed 主表。

## 当前可引用结论

最稳妥的论文主线是：

> SC-INR 将连续局部 Fourier 表达与 output cell observation 解耦，用解析 sampling response 注入 cell，在保持重建质量基本不下降的同时，改善尺度外输出采样稳定性。

当前不应写成：

> SC-INR 已经实现严格尺度等变，或已经显著超过所有 ASISR 方法。

## 结果覆盖

| 结果 | 路径 | 覆盖 | 当前用途 |
| --- | --- | --- | --- |
| 主 benchmark | `results/benchmark.json`、`results/analysis/benchmark_summary.csv` | 8 个模型，Set5/Set14/BSD100/Urban100，x2-x30，seed1 | 单 seed 主质量证据 |
| 连续尺度 | `results/continuous/*.json`、`results/analysis/figures/continuous_*.png` | x1-x30 step 0.5，seed1 | 尺度趋势诊断；正文需重画 |
| 辅助质量与一致性 | `results/analysis/seed1_aux_metrics_all8/` | BSD100/Urban100 各前 10 张，x4/x8/x16/x30，seed1 | 机制分析主证据 |
| 多 seed 当前状态 | `results/seeds/`、`results/analysis/benchmark_seed_mean_std.csv` | 当前不完整 | 不能作为公平多 seed 表 |
| 机制诊断 | `results/fce.json`、`results/phase_intervention.json` | 局部诊断 | 可作为补充，不宜主打 |

## 主 benchmark 解读

以 LTE 为参照，`SC-INR-Adaptive` 在 seed1 主 benchmark 上：

- ID 平均：`31.0515 dB`，相对 LTE `+0.0065 dB`。
- OOD 平均：`22.7471 dB`，相对 LTE `+0.0554 dB`。
- 全尺度平均：`25.5152 dB`，相对 LTE `+0.0391 dB`。

这个结果能说明 SC-INR-Adaptive 没有明显牺牲重建质量，并在 OOD scale 上有小幅改善。但提升幅度较小，必须通过多 seed、消融和机制指标支撑，不能单靠 PSNR 主表宣称强优势。

## 连续尺度解读

连续尺度曲线更适合画 `PSNR gain vs LTE`，而不是画绝对 PSNR。按 x2/x3/x4 作为近训练尺度、x6-x30 作为外推尺度，SC-INR-Adaptive 的平均 gain 为：

| 数据集 | x2/x3/x4 gain | x6-x30 gain |
| --- | ---: | ---: |
| BSD100 | `-0.0024 dB` | `+0.0304 dB` |
| Set14 | `-0.0005 dB` | `+0.0802 dB` |
| Set5 | `+0.0384 dB` | `+0.1190 dB` |
| Urban100 | `-0.0095 dB` | `+0.0574 dB` |

注意两个 caveat：

- x1 的 PSNR 异常高，且不同模型差异不适合纳入常规平均；正文应从 x2 或更高尺度开始分析。
- `results/continuous/urban100.json` 中 `LIIF-EQ` 和 `LTE-EQ` 为空，不能用它支撑完整 8 模型 Urban100 连续尺度比较。

## 辅助指标解读

`seed1_aux_metrics_all8` 是当前最强的机制证据。它回答的问题不是“谁的 benchmark 最高”，而是：

> 同一个 LR 输入下，不同输出采样尺度之间是否保持一致。

SC-INR-Adaptive 相对 LTE：

- BSD100 OOD：PSNR-Y `+0.0559 dB`，SSIM-Y `+0.00191`，texture RMSE `-0.00095`。
- Urban100 OOD：PSNR-Y `+0.0522 dB`，SSIM-Y `+0.00101`，texture RMSE `-0.00094`。
- BSD100 consistency：SC-PSNR `+8.61 dB`，texture consistency RMSE `-0.00590`。
- Urban100 consistency：SC-PSNR `+7.77 dB`，texture consistency RMSE `-0.01047`。

解释时必须保留 caveat：`LTE-NoCell` 和 `LTE-FeaturePhase` 的 consistency 也很高，主要因为它们削弱了 cell response。它们是诊断 baseline，不能仅凭 consistency 判定为更好的 ASISR 模型。

## 图表状态

适合继续打磨为正文图表：

- `seed1_aux_metrics_all8/paper/consistency_table.tex`：可作为机制分析表。
- `seed1_aux_metrics_all8/paper/quality_psnr_table.tex`：可作为辅助质量表。
- `seed1_aux_metrics_all8/paper/fig_scale_gain_vs_lte.*`：方向正确，但需要补齐 coverage 并重画。

暂不适合正文：

- `results/analysis/figures/*.png`：更像实验监控图，模型多、信息密度低。
- `fig_quality_consistency_tradeoff.*`：思路可用，但 NoCell/FeaturePhase 把坐标轴拉得过大，应拆成主模型图和诊断 baseline 补充图。
- `visual_crops/`：当前 crop 看不出关键差异，只能作为调试输出。

## 下一阶段完整工作计划

1. 多 seed：完成至少 LIIF、LTE、SC-INR-Adaptive、SC-INR-Adaptive-Signed 的公平多 seed；如果正文比较 EQ 或诊断 baseline，也要补齐相同 seed 口径。
2. Signed omega：训练完成后评估 benchmark、连续尺度、辅助质量、一致性、omega/response 分布，判断 softplus 非负频率是否造成方向性限制。
3. Sinc 因果消融：实现 `SC-INR-Adaptive w/o sinc`，保持 adaptive omega 和 decoder 结构，只令 sampling response 为 1。没有这个消融，不能说收益主要来自 sinc；如果 signed omega 成为主模型，也需要对应的 signed w/o sinc。
4. Feature-conditioned phase：尝试 `SC-INR-Adaptive + phi(z)`，但 `phi` 只能来自 feature 和采样位置，不能来自 cell/scale。该实验用于判断显式内容相位是否补充表达力，而不是恢复 LTE 的 cell-conditioned phase。
5. Rot-E 结合：先做低风险 `EQ encoder + SC-INR decoder`，验证旋转等变 encoder 与 scale-decoupled observation 是否互补；完整 equivariant omega / orientation-aware sinc 放在后续。
6. Response diagnostics：补 cell response curve、omega 分布、`omega * cell / 2` 分布、sinc attenuation 分布、负 response 比例。
7. 可视化重做：人工挑选结构纹理样例，做完整图像 + crop box + 少量关键方法 + 共享 error map 色域的正文图。
8. 论文组织：主标题和摘要使用 `sampling-consistent Fourier implicit decoding` 或 `scale-sampling consistency`，避免 `strict scale equivariance`；Method 统一 sinc 单位和频率定义；Limitations 主动说明 PSNR 提升小、不是严格尺度等变、sinc 基于 box observation 近似。

## 当前工作状态

后台训练仍在进行：

- seed2 LIIF：`save-seeds/seed2/liif/log.txt`
- seed2 LTE：`save-seeds/seed2/lte/log.txt`
- signed omega：`save/sc-inr-adaptive-signed/log.txt`

在这些训练完成前，不应更新最终多 seed 统计表。
