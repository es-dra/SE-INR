# Cell-Phase Proxy Diagnostic

本目录检查 `cell -> phase` 的显式机制路径。

## 协议

- 模型：`LTE,LTE-PhaseZ,SC-INR,SC-INR-NoSinc`
- observation scales：`4,30`，参考尺度 `x4`
- cell multipliers：`1,4`，参考 multiplier `1`

## 主要结论读法

- `LTE` 有 `phase = Linear(cell)`，因此 `cell` 改变会直接改变 Fourier-like basis phase。
- `LTE-PhaseZ`、`SC-INR`、`SC-INR-NoSinc` 没有 cell-conditioned phase，本诊断中该项应为 0。
- 本诊断只说明“是否存在移动 phase 的机制能力”，不证明一定产生视觉错误。

当前 LTE 最大 mean phase delta vs ref：`0.174740`。

## 文件

- `cell_phase_proxy.csv`
- `figures/phase_delta_by_scale.png|pdf`
- `figures/phase_delta_by_multiplier.png|pdf`
- `run_config.json`
