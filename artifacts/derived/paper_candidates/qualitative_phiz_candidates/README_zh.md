# SC-INR qualitative candidate crops

本目录是最终候选 `SC-INR` 的论文视觉图候选，不是最终正文图。其 raw/checkpoint
兼容名为 `SC-INR+PhiZ`。候选由 `scripts/prepare_qualitative_figure.py` 生成，
协议是当前 benchmark 使用的 HR 现场 bicubic downsampling。

## 生成协议

- 方法：`GT`、`Bicubic`、`LIIF`、`LTE`、`SC-INR-NoPhi`、`SC-INR`
- 目标模型：`SC-INR`（raw key `SC-INR+PhiZ`）
- 参照模型：`LTE`
- crop 大小：`96 x 96`
- 初筛：`--auto_delta_crop --target_model SC-INR --baseline_model LTE --min_texture_quantile 0.50`
- 输出：
  - 每个候选的 `.png/.pdf`：full image + crop rectangle + method crops + error maps
  - 每个候选的 `_metrics.csv`：局部 crop PSNR-Y/RMSE-Y
  - 每个候选的 `_crop_candidates.csv`：按局部增益排序的候选 crop
  - `candidate_summary.csv`：所有已生成候选的汇总
  - `contact_sheet.png`：所有候选 crop 的快速人工审查图

注意：部分既有图片是在展示命名调整前生成的，图中可能仍显示旧标签
`SC-INR` / `SC-INR+PhiZ`。按当前命名应分别理解为 `SC-INR-NoPhi` /
最终候选 `SC-INR`。正式论文图应重新导出，避免图例歧义。

## 候选排序摘要

| Candidate | Crop | PhiZ-LTE | PhiZ-SC-INR | Comment |
| --- | ---: | ---: | ---: | --- |
| `urban100_img012_x8_delta_phiz_vs_lte` | `(432,864)` | `+2.78` | `+3.61` | 建筑重复线条，局部增益大；需人工确认是否错频更少。 |
| `urban100_img004_x8_delta_phiz_vs_lte` | `(144,912)` | `+2.46` | `+0.64` | 孔洞重复纹理，最接近 Rot-E 式视觉例子。 |
| `urban100_img018_x8_delta_phiz_vs_lte` | `(48,384)` | `+2.29` | `-0.00` | 相对 LTE 明显，但与 SC-INR 基本相同；不适合证明 PhiZ 超过 SC-INR。 |
| `set14_ppt3_x8_delta_phiz_vs_lte` | `(96,240)` | `+2.12` | `+1.57` | 文字/边缘恢复，可能适合展示结构清晰度。 |
| `urban100_img004_x16_delta_phiz_vs_lte` | `(528,384)` | `+1.94` | `+1.51` | x16 外推候选，可用于展示大尺度时局部稳定性。 |
| `set14_barbara_x16_delta_phiz_vs_lte` | `(48,144)` | `+1.06` | `+0.47` | 经典纹理图，但视觉差异可能不如数字明显。 |
| `set14_zebra_x8_delta_phiz_vs_lte` | `(48,192)` | `+0.53` | `+0.03` | 相对 SC-INR 几乎不变，更适合作为中性例。 |
| `set14_barbara_x8_delta_phiz_vs_lte` | `(96,240)` | `+0.45` | `+0.22` | 中等增益候选。 |
| `set14_comic_x8_delta_phiz_vs_lte` | `(144,96)` | `+0.14` | `+0.34` | 小增益候选，更适合补充或中性例。 |

## 审查意见

- `--auto_delta_crop` 会倾向于挑出局部最大收益，存在 cherry-pick 风险；这些图只能作为候选探索产物。
- 正式论文图建议从固定候选池中人工确认，并同时保留优势样例、中性样例和至少一个不明显样例。
- 如果正文只放优势图，补充材料应提供 `candidate_summary.csv`、若干非优势候选和筛选协议，避免视觉证据被审稿人认为不透明。
- 最稳妥的正文组合：一个局部恢复图 + 一个 same-LR cross-scale consistency difference map。前者回答视觉质量，后者回答本方法区别于普通 PSNR 提升的机制。

## 推荐优先人工查看

1. `contact_sheet.png`
2. `urban100_img004_x8_delta_phiz_vs_lte.png`
3. `urban100_img012_x8_delta_phiz_vs_lte.png`
4. `set14_ppt3_x8_delta_phiz_vs_lte.png`
5. `urban100_img004_x16_delta_phiz_vs_lte.png`
