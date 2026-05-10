# Response/Omega Diagnostics 2026-05-10

本目录补充 response/omega 机制诊断，用于解释为什么
`SC-INR-NoSinc` 在 same-LR self-consistency 指标上显著高于完整 `SC-INR`。
核心问题不是重算 benchmark，而是区分：

- point/self-consistency：模型对 cell 不敏感，因此跨 cell 输出一致；
- footprint observation correctness：cell 作为像素 footprint，通过 analytic
  response 改变 Fourier observation。

## 协议

- response/omega 分布模型：`SC-INR-NoPhi`, `SC-INR-NoPhi-Signed`, `SC-INR`,
  `SC-INR-NoSinc`
- response/omega 数据：BSD100 / Urban100 sorted 前 5 张
- response/omega 观测尺度：x4, x8, x16, x30
- cell-only 曲线模型：`LTE`, `SC-INR-NoPhi`, `SC-INR-NoPhi-Signed`, `SC-INR`,
  `SC-INR-NoSinc`
- cell-only 数据：BSD100 / Urban100 sorted 前 5 张
- cell-only 输入：同一 x4 LR、同一 query coordinate，只改变 cell scale
- cell-only scales：x4, x6, x8, x12, x16, x24, x30
- query sampling：每张图最多 4096 个 query points

命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python scripts/analysis/analyze_sampling_response.py \
  --mode both \
  --models SC-INR-NoPhi,SC-INR-NoPhi-Signed,SC-INR,SC-INR-NoSinc \
  --datasets bsd100,urban100 \
  --max_images 5 \
  --lr_scale 4 \
  --scales 4,8,16,30 \
  --cell_models LTE,SC-INR-NoPhi,SC-INR-NoPhi-Signed,SC-INR,SC-INR-NoSinc \
  --cell_datasets bsd100,urban100 \
  --cell_image '' \
  --cell_max_images 5 \
  --cell_lr_scale 4 \
  --cell_ref_scale 4 \
  --cell_scales 4,6,8,12,16,24,30 \
  --max_queries 4096 \
  --eval_bsize 50000 \
  --device cuda:0 \
  --out artifacts/derived/analysis/response_omega_diagnostics_2026-05-10
```

`cuda:0` 是进程内可见设备；物理绑定为 GPU 2。

## 覆盖

- `response_distribution.csv`：160 条数据行。
- `response_distribution_summary.csv`：16 条数据行。
- `cell_response_curve.csv`：350 条数据行。
- `cell_response_curve_summary.csv`：35 条数据行。
- `cell_response_curve_by_dataset.csv`：70 条数据行。

## Key Results

### Active Response

`response_*` 表示由 learned omega 计算出的 analytic sinc response。
`active_response_*` 表示模型实际使用的 response；对 `SC-INR-NoSinc`，它恒为 1。

Active attenuation mean，越大表示 cell 对 Fourier observation 的实际衰减越强：

| Model | x4 | x8 | x16 | x30 |
| --- | ---: | ---: | ---: | ---: |
| `SC-INR-NoPhi` | 0.037106 | 0.009583 | 0.002415 | 0.000688 |
| `SC-INR-NoPhi-Signed` | 0.036853 | 0.009494 | 0.002392 | 0.000681 |
| `SC-INR` | 0.040495 | 0.010570 | 0.002672 | 0.000762 |
| `SC-INR-NoSinc` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

`SC-INR-NoSinc` 的 learned omega 仍然可以产生类似的 hypothetical analytic
attenuation，例如 x4 的 `analytic_attenuation_mean` 为 0.037540；但该 response
在模型中没有被使用。

### Cell-Only Sensitivity

同一 x4 LR、同一 query coordinate，只改变 cell scale，计算输出 Y 通道相对
x4 cell 的 RMSE：

| Model | x8 | x16 | x30 | x8/x16/x30 avg |
| --- | ---: | ---: | ---: | ---: |
| `LTE` | 0.005267 | 0.008329 | 0.010233 | 0.007943 |
| `SC-INR-NoPhi` | 0.002862 | 0.003462 | 0.003598 | 0.003308 |
| `SC-INR-NoPhi-Signed` | 0.002662 | 0.003192 | 0.003310 | 0.003055 |
| `SC-INR` | 0.002854 | 0.003395 | 0.003515 | 0.003255 |
| `SC-INR-NoSinc` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

`SC-INR-NoSinc` 对 cell 完全不敏感，这解释了它在 same-LR self-consistency
指标上的异常高分：它不是更正确地建模 footprint，而是没有实际的 cell-dependent
observation response。

### Omega Notes

- `SC-INR` signed omega negative component fraction: 0.3951。
- `SC-INR-NoSinc` signed omega negative component fraction: 0.3960。
- `SC-INR-NoPhi-Signed` signed omega negative component fraction: 0.3648。
- `SC-INR-NoPhi` 使用旧 softplus-positive omega，negative fraction 为 0。
- `SC-INR-NoSinc` near-bound fraction 为 0.0135，高于 `SC-INR` 的 0.000084；
  这是一个值得后续监控的优化差异，但当前不能直接解释为好坏。

## Interpretation

- 该诊断支持上一轮的 caveat：same-LR self-consistency 会奖励 cell-independent
  decoder，不能单独证明 sampling-consistent observation。
- 完整 `SC-INR` 和 no-phase SC-INR 变体确实使用了非零 analytic response；
  NoSinc 虽然学到类似 omega，但实际 active response 恒为 1。
- LTE 的 cell-only sensitivity 最大，符合其 learned cell-conditioned phase
  会强烈改变输出函数的机制风险。
- `SC-INR` 的 cell-only sensitivity 明显小于 LTE，但非零；这更接近
  scale-decoupled analytic observation，而不是完全忽略 cell。

## Paper Boundary

可写：

- NoSinc 的高 same-LR consistency 来自 cell-insensitive behavior，不能作为
  更正确 observation modeling 的证据。
- SC-INR 的 decoder 使用非零 analytic response，因此与 NoSinc 的 zero
  cell-sensitivity 有机制差别。

不可写：

- 该诊断证明 sinc 是唯一因果。
- 高 same-LR consistency 等价于正确 sampling consistency。
- 当前小样本 response/omega 诊断替代 full benchmark 或 multi-seed 结论。
