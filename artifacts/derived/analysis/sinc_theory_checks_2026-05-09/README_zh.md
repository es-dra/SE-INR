# Sinc 理论支撑检查

本目录保存两个轻量理论支撑实验，用于说明 SC-INR 的 analytic sinc
observation response 在实现层面是可验证、可诊断的。它们不是 benchmark，
也不能单独证明 sinc 是性能提升的唯一原因。

## 1. Fourier footprint 平均 property test

路径：`property_test/`

命令：

```bash
python scripts/analysis/verify_sinc_observation.py \
  --out artifacts/derived/analysis/sinc_theory_checks_2026-05-09/property_test \
  --num_trials 2000 \
  --quad_order 64 \
  --seed 1
```

结果：

- 随机 2D Fourier component trials：`2000`
- cos/sin rows：`4000`
- 最大绝对误差：`5.329071e-15`
- p99 绝对误差：`3.441691e-15`
- 平均绝对误差：`8.843487e-16`

解释：在数值积分精度内，矩形 footprint 上的 Fourier average 与
`sinc(omega_h * cell_h / 2) * sinc(omega_w * cell_w / 2)` 解析响应一致。

## 2. 已训练模型 response distribution

路径：`response_diagnostics/`

命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python scripts/analysis/analyze_sampling_response.py \
  --mode response \
  --models SC-INR-NoPhi-Signed,SC-INR \
  --datasets bsd100 \
  --max_images 1 \
  --lr_scale 4 \
  --scales 4,8,16,30 \
  --device cuda:0 \
  --out artifacts/derived/analysis/sinc_theory_checks_2026-05-09/response_diagnostics
```

关键观察：

| Model | Scale | Response mean | Response q05 | Response q50 | Response q95 | omega negative fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `SC-INR-NoPhi-Signed` | x4 | 0.961223 | 0.817539 | 0.986701 | 0.999893 | 0.361276 |
| `SC-INR-NoPhi-Signed` | x30 | 0.999283 | 0.996564 | 0.999763 | 0.999998 | 0.361276 |
| `SC-INR` | x4 | 0.959919 | 0.805194 | 0.989624 | 0.999938 | 0.396298 |
| `SC-INR` | x30 | 0.999246 | 0.996316 | 0.999815 | 0.999999 | 0.396298 |

解释：在当前 learned omega 范围下，低倍率较大的 observation cell 有更明显的
frequency attenuation；高倍率较小 cell 的 response 更接近 1。最终 `SC-INR`
保留 signed omega 分布和 analytic response，且没有出现大量负 response。

## 解释边界

- 这些检查支持“SC-INR 的 observation formula 是数学和实现一致的”。
- 这些检查不证明 benchmark gain 一定由 sinc 导致；这个问题仍依赖
  `SC-INR-NoSinc` 消融结果。
- response distribution 只跑了 BSD100 首图，用于机制可视化和 sanity check；
  不能当成完整数据集统计。
