# Selected Qualitative Figures Seed1

本目录将用户确认的两张 PhiZ qualitative 候选图重新导出为正式论文候选图。
与旧候选目录不同，这里使用当前展示命名：

- `SC-INR-NoPhi`: 旧 no-phase 主线。
- `SC-INR`: final candidate, signed omega + feature-conditioned phase + sinc。

## 协议

- 数据：Urban100 HR，按 benchmark 协议现场 bicubic downsampling。
- scale：x8。
- 方法列：`GT`, `Bicubic`, `LTE`, `SC-INR-NoPhi`, `SC-INR`。
- 附加：error maps、crop PSNR-Y。
- 图像：
  - `img_004.png`, crop `(y=144, x=912, size=96)`。
  - `img_012.png`, crop `(y=432, x=864, size=96)`。
- 生成脚本：`scripts/viz/prepare_qualitative_figure.py`。
- 设备：物理 GPU 1，经 `CUDA_VISIBLE_DEVICES=1` 绑定；配置文件内记录为进程内
  `cuda:0`。

## 命令

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python scripts/viz/prepare_qualitative_figure.py \
  --dataset urban100 --image img_004.png --scale 8 \
  --models Bicubic,LTE,SC-INR-NoPhi,SC-INR \
  --crop 144,912,96 \
  --with_error_maps --show_crop_psnr \
  --device cuda:0 --eval_bsize 50000 \
  --out artifacts/derived/paper_figures/qualitative_selected_seed1 \
  --name urban100_img004_x8_selected_gt_bicubic_lte_nophi_scinr

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python scripts/viz/prepare_qualitative_figure.py \
  --dataset urban100 --image img_012.png --scale 8 \
  --models Bicubic,LTE,SC-INR-NoPhi,SC-INR \
  --crop 432,864,96 \
  --with_error_maps --show_crop_psnr \
  --device cuda:0 --eval_bsize 50000 \
  --out artifacts/derived/paper_figures/qualitative_selected_seed1 \
  --name urban100_img012_x8_selected_gt_bicubic_lte_nophi_scinr
```

## Local Crop Metrics

`urban100_img004_x8`:

| Model | PSNR-Y | RMSE-Y |
| --- | ---: | ---: |
| Bicubic | 16.2966 | 0.152540 |
| LTE | 19.0226 | 0.111345 |
| SC-INR-NoPhi | 20.8467 | 0.092965 |
| SC-INR | 21.4880 | 0.087662 |

`urban100_img012_x8`:

| Model | PSNR-Y | RMSE-Y |
| --- | ---: | ---: |
| Bicubic | 18.8876 | 0.110371 |
| LTE | 20.4458 | 0.093134 |
| SC-INR-NoPhi | 19.6163 | 0.101220 |
| SC-INR | 23.2229 | 0.069400 |

## 解释边界

- 这些图是 selected qualitative examples，只能说明局部候选区域中 `SC-INR`
  恢复更清晰或局部 PSNR 更高。
- 不能据此宣称 average visual quality 已经整体更好。
- 如果正文使用优势图，建议补充材料保留候选筛选协议和非优势候选，以降低
  cherry-pick 风险。
