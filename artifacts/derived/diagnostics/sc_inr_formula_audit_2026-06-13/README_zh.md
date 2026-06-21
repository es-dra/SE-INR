# SC-INR 公式叙事与 x30 定性图复核 2026-06-13

本目录保存一次小规模复核产物，用于辅助解释 `Urban100/img_016.png x30 crop=(672,384,96)` 中
LIIF/LTE 输出结构扭曲的问题。该目录不是新的主 benchmark，也不作为论文主指标。

## 产物

- `urban100_img016_x30_crop21_reproduced_panel.png`：CPU 复现的 crop 面板，只用于确认保存图不是可视化拼接错误。
- `urban100_img016_x30_crop21_structure_check.csv`：未裁边 crop 上的非正式方向/结构检查。
- `urban100_img016_x30_crop21_structure_check_shaved.csv`：按原局部 PSNR 裁边口径重新计算的非正式方向/结构检查。
- `urban100_img016_x30_crop21_pairwise_output_similarity.csv`：同一 crop 上各输出之间的
  Y 通道 pairwise RMSE/PSNR/correlation，用于确认 LIIF/LTE 视觉相近是否客观存在。

## 使用边界

- 正式数值仍以
  `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_region_pool.csv`
  和原图为准。
- 方向/结构检查只是帮助理解“输出条纹方向是否发生漂移”，不是新主指标；局部裁边后该指标对单条纹位置很敏感，不应升级为论文 claim。
- pairwise similarity 只描述该 selected crop 的输出相似性，不能外推到全数据集。
- 该例子来自按 `min(SC-INR-LIIF, SC-INR-LTE)` 排序的 OOD selected advantage region，只能说明强优势候选区域，不能代表平均视觉质量。
