# SC-INR 证据链补强诊断 2026-05-21

本目录当前主用途是定位 `SC-INR` 相对外部基线 `LIIF/LTE` 的可视优势区间。
消融相关诊断默认不再生成；如需内部审查，可显式开启 `--make_internal_plots`。

## 协议

- 逐图统计：models `LIIF,LTE,SC-INR`，seeds `1,2,3`，datasets `set5,set14,bsd100,urban100`，scales `2,3,4,6,8,12,16,24,30`。
- 局部候选池：models `Bicubic,LIIF,LTE,SC-INR,SC-INR-NoPhi,SC-INR-NoSinc`，datasets `bsd100,urban100`，scales `4,8,16,30`。
- crop 规则：固定 `96x96`，stride `96`，每图每尺度最多 `32` 个 crop。
- crop 分层只依赖 GT：texture local variance、Sobel edge、highpass energy；不按模型胜负筛选。
- 论文候选图只从 OOD 且非平坦结构 crop 中选择，排序指标为 `min(SC-INR-LIIF, SC-INR-LTE)`。
- 主图只展示 `GT/Bicubic/LIIF/LTE/SC-INR`；NoPhi/NoSinc 留作后续单独消融讨论。
- bootstrap CI 按 seed+dataset+image 聚类，避免把同一图多个 scale 当完全独立样本。

## 关键结果摘要

### 3-seed paired statistics

| Baseline | Split | Mean Δ | Median Δ | Win-rate | 95% CI mean |
| --- | --- | ---: | ---: | ---: | --- |
| LIIF | ALL | 0.0333 | 0.0210 | 0.671 | [0.0189, 0.0467] |
| LIIF | OOD | 0.0291 | 0.0216 | 0.689 | [0.0197, 0.0389] |
| LTE | ALL | 0.0224 | 0.0170 | 0.638 | [0.0070, 0.0356] |
| LTE | OOD | 0.0434 | 0.0320 | 0.746 | [0.0325, 0.0535] |

### 外部基线优势区间

这里的 `win external` 表示同一个 crop 上 SC-INR 同时优于 LIIF 和 LTE。
注意：区域中位数仍可能被未同时赢两个模型的样本拉低；因此论文展示应绑定
完整候选池统计、top/neutral/failure 案例，而不是只展示 top crop。

| Scale | n | median vs LIIF | median vs LTE | win external | q75 min external Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| x16 | 1560 | 0.0276 | 0.0394 | 0.457 | 0.0837 |
| x30 | 1560 | 0.0086 | 0.0428 | 0.417 | 0.0657 |
| x8 | 1560 | 0.0332 | 0.0279 | 0.454 | 0.0632 |

### 论文定性图候选

`external_advantage_region_pool.csv` 按外部基线优势排序，并保留
top/neutral/failure。当前已人工筛选保留下来的主候选在
`figures/external_advantage_crops/`；更适合正文展示的大上下文版本在
`figures/external_advantage_context_zoom/`。若重新运行 `render_external`，
脚本会重新生成带 `_clean.png` 后缀的纯 crop panel；当前目录内容以实际保留文件为准。

| Type | Dataset/Image | Scale | Crop | bins | min Δ vs LIIF/LTE |
| --- | --- | --- | --- | --- | ---: |
| top | Urban100/img_007.png | x30 | 3@(0,672) | high/low/high | 10.026 |
| top | Urban100/img_016.png | x30 | 21@(672,384) | high/high/high | 7.958 |
| top | Urban100/img_011.png | x8 | 12@(384,96) | mid/mid/mid | 4.663 |
| top | Urban100/img_011.png | x16 | 18@(576,192) | mid/mid/low | 4.226 |
| top | BSD100/167062.png | x30 | 12@(192,192) | high/mid/high | 3.999 |
| top | BSD100/167062.png | x30 | 17@(204,192) | high/mid/high | 3.303 |
| top | Urban100/img_001.png | x30 | 8@(96,864) | high/mid/high | 3.055 |
| top | Urban100/img_016.png | x30 | 22@(672,768) | high/high/high | 2.814 |
| top | BSD100/101087.png | x16 | 1@(0,96) | high/low/high | 2.310 |
| top | Urban100/img_021.png | x30 | 20@(768,288) | low/mid/mid | 1.963 |
| top | Urban100/img_002.png | x30 | 0@(0,0) | high/mid/mid | 1.962 |
| top | Urban100/img_007.png | x30 | 2@(0,480) | high/low/high | 1.608 |

目前最适合正文的叙述是：在预注册候选池里，SC-INR 相对 LIIF/LTE 的
明显视觉优势主要出现在 OOD 建筑/纹理/高频结构 crop，尤其是 Urban100 的
x8/x16/x30 局部结构区域；但总体外部 win rate 仍不到 50%，所以这些是
selected advantage regions，不是全局普遍优势。

### x4 sanity / 边界对照

x4 属于当前训练/ID 尺度范围，不是本文主张的 OOD 优势区间。这里保留少量
x4 neutral/sanity crop，用来说明低倍率下各方法差距通常很小，避免正文只展示
x8-x30 时被误解为刻意隐藏低倍率结果。

| Dataset/Image | Crop | bins | min Δ vs LIIF/LTE |
| --- | --- | --- | ---: |
| Urban100/img_018.png | 6@(96,576) | high/mid/high | -0.0001 |
| BSD100/145086.png | 10@(192,0) | low/mid/mid | -0.0002 |
| BSD100/159008.png | 14@(192,384) | high/high/high | 0.0004 |
| Urban100/img_006.png | 1@(0,384) | high/high/high | 0.0005 |

## 产物

- `per_image_per_scale_metrics.csv`：逐 seed / image / scale / model 的 PSNR-Y 等质量指标；本地可再生成底层表，默认不进源码管理。
- `paired_delta_per_image_scale.csv`：SC-INR 相对 LIIF/LTE 的逐样本 delta；本地可再生成底层表，默认不进源码管理。
- `paired_delta_*summary*.csv` 和 `paired_delta_bootstrap_ci.csv`：统计摘要和 cluster bootstrap CI。
- `local_crop_descriptors.csv`：只由 GT 决定的 crop 坐标和 texture/edge/highpass 分层；本地可再生成底层表，默认不进源码管理。
- `local_crop_metrics.csv`：每个 crop 的模型质量指标；本地可再生成底层表，默认不进源码管理。
- `external_advantage_per_crop.csv`：每个 crop 上 SC-INR 相对 LIIF/LTE 的外部优势。
- `external_advantage_region_pool.csv`：论文定性图候选，含 top/neutral/failure。
- `figures/external_advantage_crops/`：用户筛选后保留的正文/appendix 候选 crop panel；目录内容以实际文件为准，默认只本地保留。
- `figures/external_advantage_context_zoom/`：围绕已筛选外部优势图生成的更大上下文 + zoom panel，优先用于正文候选，默认只本地保留。
- `external_advantage_x4_sanity_pool.csv` 和 `figures/external_advantage_x4_sanity/`：x4 边界对照，不作为主优势证据；图目录默认只本地保留。
- `local_crop_paired_deltas.csv`、`local_delta_summary_*`、`advantage_zone_*`、`advantage_margin_*` 仅在显式内部诊断模式下生成。

## 使用边界

- 该目录不能证明 strict scale equivariance、sinc 唯一因果或 exact RGB box integral。
- 正文图只能说明 selected external-baseline advantage regions，必须和候选池统计一起解释。
- x4 sanity 只用于展示 ID 低倍率边界，不能写成 SC-INR 在 x4 也有明显视觉优势。
- 当前结果不支持“SC-INR 在所有局部区域普遍优于 LIIF/LTE”的强说法。
- 消融比较后续单独讨论；本 README 的主展示不把 NoPhi/NoSinc 作为视觉主线。
- 如果 win-rate 接近 50% 或 bootstrap CI 跨 0，应收缩为 modest average gain / diagnostic evidence。
- same-LR consistency 不在本目录中作为 footprint correctness 主证据。
