# Effective Amplitude Diagnostics

本目录诊断 SC-INR decoder 输入层面的 effective amplitude：

`q_coef(z) * W(omega(z), c)`。

这里的 `q_coef` 是送入 MLP 前、分别乘到 cos/sin 通道的 feature-level coefficient；
脚本用每个频率的 cos/sin coefficient pair magnitude 作为 `A(z)` 的可解释 proxy。
因此本结果不能解释为最终 RGB 图像频谱的严格振幅，原因是后续还有 MLP、local ensemble
和 residual upinput。

## 文件

- `effective_amplitude_stats.csv`：逐模型、数据集、图像、scale、频率分组的统计。
- `effective_amplitude_summary.csv`：按模型、scale、频率分组聚合。
- `figures/active_response_abs_by_freq.png`：不同频率组的 `|W|` 曲线。
- `figures/effective_pair_mag_by_freq.png`：不同频率组的 effective coefficient magnitude。
- `figures/effective_energy_ratio_by_freq.png`：`A_eff` 相对 `A` 的能量比例。
- `run_config.json`：运行参数。

## 覆盖范围

- models: `SC-INR,SC-INR-NoSinc`
- datasets: `bsd100,urban100`
- max_images: `3`
- max_queries: `1024`
- scales: `2,4,8,16,30`

如果这里是小样本配置或单图配置，本结果只能作为 mechanism sanity check，不能作为
数据集级趋势、模型排序或多 seed 证据。

## 解释边界

- `response` 是 signed sinc response；它可能为负，负号可视作 Fourier 分量的相位翻转。
- `active_response` 是模型实际使用的 response；对 `SC-INR-NoSinc` 它恒为 1。
- `effective_pair_mag` 使用 magnitude，因此反映的是输入强度变化，不保留 signed response 的符号。
- 本诊断是 `diagnostic-only`，需要与 benchmark、NoSinc 负控和 cell-intervention 一起解释。

## 参数

```json
{
  "self_test": false,
  "out": "artifacts/derived/diagnostics/incode_method_support_2026-06-20/effective_amplitude",
  "device": "cpu",
  "models": "SC-INR,SC-INR-NoSinc",
  "datasets": "bsd100,urban100",
  "max_images": 3,
  "lr_scale": 4,
  "scales": "2,4,8,16,30",
  "max_queries": 1024
}
```
