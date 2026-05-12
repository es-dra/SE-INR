# Final SC-INR Seed1 Auxiliary Metrics

本目录补充最终候选 `SC-INR` 的 seed1 auxiliary metrics。结果用于判断
feature-conditioned phase (`phi(z)`) 是否保持 `SC-INR-NoPhi` 的
decoder-side consistency 优势，并与 `LTE`、`SC-INR-NoPhi`、
`SC-INR-NoPhi-Signed` 对照。

## 协议

- 模型：`LTE`, `SC-INR-NoPhi`, `SC-INR-NoPhi-Signed`, `SC-INR`
- 数据：BSD100 / Urban100 sorted 前 10 张
- 质量尺度：x4, x8, x16, x30
- OOD 汇总：x8, x16, x30
- consistency：same-LR x8/x16/x30 -> x4
- 输出：`quality_metrics.csv`, `quality_summary.csv`,
  `consistency_metrics.csv`, `consistency_summary.csv`
- 命令：

```bash
python scripts/analysis/evaluate_seed1_aux_metrics.py \
  --models LTE,SC-INR-NoPhi,SC-INR-NoPhi-Signed,SC-INR \
  --datasets bsd100,urban100 \
  --scales 4,8,16,30 \
  --consistency_pairs '8,4;16,4;30,4' \
  --max_images 10 \
  --device cuda:1 \
  --eval_bsize 50000 \
  --out artifacts/derived/diagnostics/sc_inr_final_aux_metrics_seed1 \
  --skip_visuals
```

## 关键结果

OOD quality PSNR-Y:

| Dataset | `SC-INR - LTE` | `SC-INR - SC-INR-NoPhi` | `SC-INR - SC-INR-NoPhi-Signed` |
| --- | ---: | ---: | ---: |
| BSD100 | +0.0506 dB | -0.0053 dB | +0.0043 dB |
| Urban100 | +0.0746 dB | +0.0223 dB | +0.0327 dB |

OOD texture RMSE-Y delta（负数更好）：

| Dataset | `SC-INR - LTE` | `SC-INR - SC-INR-NoPhi` | `SC-INR - SC-INR-NoPhi-Signed` |
| --- | ---: | ---: | ---: |
| BSD100 | -0.001063 | -0.000112 | -0.000197 |
| Urban100 | -0.001273 | -0.000332 | -0.000377 |

Same-LR consistency PSNR-Y:

| Dataset | LTE | SC-INR-NoPhi | SC-INR-NoPhi-Signed | SC-INR |
| --- | ---: | ---: | ---: | ---: |
| BSD100 | 44.1914 | 52.8053 | 52.9039 | 52.1421 |
| Urban100 | 39.2979 | 47.0694 | 48.1003 | 47.1128 |

Consistency delta:

| Dataset | `SC-INR - LTE` | `SC-INR - SC-INR-NoPhi` | `SC-INR - SC-INR-NoPhi-Signed` |
| --- | ---: | ---: | ---: |
| BSD100 | +7.9508 dB | -0.6632 dB | -0.7618 dB |
| Urban100 | +7.8149 dB | +0.0434 dB | -0.9875 dB |

## 解释边界

- `SC-INR` 相比 `LTE` 保持了强 consistency 优势，因此当前没有看到 PhiZ
  破坏 decoder-side sampling consistency 的证据。
- `SC-INR` 不是 consistency 最强变体：BSD100 上低于 `SC-INR-NoPhi` 和
  `SC-INR-NoPhi-Signed`，Urban100 上与 `SC-INR-NoPhi` 基本持平但低于
  `SC-INR-NoPhi-Signed`。
- 质量端更支持最终候选：`SC-INR` 在 Urban100 OOD 上相对所有三个对照均提升，
  BSD100 OOD 相对 `SC-INR-NoPhi` 有极小回退但 texture RMSE 仍略好。
- 这些是 seed1、每数据集前 10 张的 auxiliary evidence，不能替代 full benchmark
  或 multi-seed 结论。
