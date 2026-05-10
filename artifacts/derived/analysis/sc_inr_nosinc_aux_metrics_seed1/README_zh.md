# SC-INR-NoSinc Seed1 Auxiliary Metrics

本目录补充 `SC-INR-NoSinc` 的 seed1 auxiliary metrics，用于检查移除 analytic
sinc response 后，同一协议下的重建质量、纹理误差和 same-LR cross-scale
self-consistency 是否相对完整 `SC-INR` 退化。

## 协议

- 模型：`LTE`, `SC-INR-NoPhi`, `SC-INR-NoPhi-Signed`, `SC-INR`, `SC-INR-NoSinc`
- quality 额外包含：`Bicubic`
- 数据：BSD100 / Urban100 sorted 前 10 张
- 质量尺度：x4, x8, x16, x30
- OOD 汇总：x8, x16, x30
- consistency：same-LR x8/x16/x30 -> x4
- 输出：`quality_metrics.csv`, `quality_summary.csv`,
  `consistency_metrics.csv`, `consistency_summary.csv`
- 校验：共享模型与
  `artifacts/derived/analysis/sc_inr_final_aux_metrics_seed1/` 的 summary 行完全一致。

运行命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python scripts/analysis/evaluate_seed1_aux_metrics.py \
  --models LTE,SC-INR-NoPhi,SC-INR-NoPhi-Signed,SC-INR,SC-INR-NoSinc \
  --datasets bsd100,urban100 \
  --scales 4,8,16,30 \
  --consistency_pairs '8,4;16,4;30,4' \
  --max_images 10 \
  --device cuda:0 \
  --eval_bsize 50000 \
  --out artifacts/derived/analysis/sc_inr_nosinc_aux_metrics_seed1 \
  --skip_visuals
```

`run_config.json` 中记录的 `cuda:0` 是进程内可见设备；物理绑定为 GPU 2。

## 覆盖校验

- `quality_metrics.csv`：480 条数据行。
- `quality_summary.csv`：48 条数据行。
- `consistency_metrics.csv`：300 条数据行。
- `consistency_summary.csv`：30 条数据行。
- 预期覆盖为 `(Bicubic + 5 models) x 2 datasets x 10 images x 4 scales`
  和 `5 models x 2 datasets x 10 images x 3 pairs`，行数匹配。

## OOD Quality

OOD PSNR-Y，按 x8/x16/x30 平均：

| Dataset | LTE | SC-INR-NoPhi | SC-INR-NoPhi-Signed | SC-INR | SC-INR-NoSinc |
| --- | ---: | ---: | ---: | ---: | ---: |
| BSD100 | 22.8065 | 22.8624 | 22.8528 | 22.8571 | 22.8495 |
| Urban100 | 20.5118 | 20.5641 | 20.5537 | 20.5864 | 20.6072 |
| All | 21.6592 | 21.7133 | 21.7033 | 21.7218 | 21.7284 |

`SC-INR-NoSinc - SC-INR`:

| Dataset | OOD PSNR-Y | OOD texture RMSE-Y | OOD highpass RMSE-Y |
| --- | ---: | ---: | ---: |
| BSD100 | -0.0076 dB | -0.000037 | +0.000102 |
| Urban100 | +0.0208 dB | -0.000383 | -0.000003 |
| All | +0.0066 dB | -0.000210 | +0.000049 |

该小子集上的 OOD auxiliary quality 不显示 NoSinc 相对 `SC-INR` 明显退化；
这与 full benchmark 中 `SC-INR-NoSinc` OOD 相对完整 `SC-INR` 低 `0.0467 dB`
并不矛盾，因为 auxiliary 只覆盖 BSD100/Urban100 前 10 张和 Y 通道派生指标。

## Same-LR Consistency

Consistency PSNR-Y，按 x8->x4、x16->x4、x30->x4 平均：

| Dataset | LTE | SC-INR-NoPhi | SC-INR-NoPhi-Signed | SC-INR | SC-INR-NoSinc |
| --- | ---: | ---: | ---: | ---: | ---: |
| BSD100 | 44.1914 | 52.8053 | 52.9039 | 52.1421 | 69.0870 |
| Urban100 | 39.2979 | 47.0694 | 48.1003 | 47.1128 | 64.6241 |
| All | 41.7447 | 49.9374 | 50.5021 | 49.6275 | 66.8555 |

`SC-INR-NoSinc - SC-INR` consistency delta：

| Dataset | PSNR-Y | RMSE-Y | Texture RMSE-Y |
| --- | ---: | ---: | ---: |
| BSD100 | +16.9448 dB | -0.002240 | -0.003744 |
| Urban100 | +17.5112 dB | -0.004057 | -0.007312 |
| All | +17.2280 dB | -0.003149 | -0.005528 |

## 解释边界

- 当前结果不支持“移除 sinc 会破坏 same-LR self-consistency”这个说法。
  相反，`SC-INR-NoSinc` 在该指标上显著更高。
- 这并不说明 NoSinc 是更好的 sampling-consistent observation model。更合理的解释是：
  关闭 sinc 后，decoder 几乎失去 cell-dependent analytic observation response，
  因而更接近 cell-independent point-function；当前 same-LR 指标会奖励这种
  自一致性，但它没有检验像素 footprint 的解析观测是否正确。
- 因此，NoSinc auxiliary metrics 暴露了一个诊断限制：same-LR consistency
  必须和 fidelity、texture/highpass、full benchmark、以及更直接的 observation
  response 诊断一起解释，不能单独作为 sinc 机制证据。
- paper-level 表述应保持克制：full benchmark 提示 sinc 对最终候选 PSNR 有贡献；
  但当前 auxiliary consistency 不支持“sinc 提升 same-LR self-consistency”，也
  不能证明 sinc 是唯一因果。
