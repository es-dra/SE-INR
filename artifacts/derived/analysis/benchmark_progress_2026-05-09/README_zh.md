# 2026-05-09 benchmark 进展汇总

本目录整理两个新增 seed1 变体和核心三模型多 seed benchmark 的阶段性结果。这里的表格是从 raw JSON 派生出来的，不覆盖主 benchmark。

## 原始结果来源

- `results/benchmark_seed1_with_signed_phiz.json`
  - seed1，包含 `SC-INR-Signed` 与 `SC-INR+PhiZ`。
  - 评估命令：`python eval_full.py --device 1 --save_root ./save --models 'SC-INR-Signed,SC-INR+PhiZ' --output results/benchmark_seed1_with_signed_phiz.json --skip_existing`
- `results/seeds/benchmark_seed2.json`
  - seed2，包含 `LIIF`、`LTE`、`SC-INR-Adaptive`。汇总时将 `SC-INR-Adaptive` 作为论文展示名 `SC-INR`。
- `results/seeds/benchmark_seed3.json`
  - seed3，已补齐 `LTE`，当前包含 `LIIF`、`LTE`、`SC-INR`。
  - LTE 补评命令：`python eval_full.py --device 1 --save_root ./save-seeds/seed3 --models LTE --output ./results/seeds/benchmark_seed3.json --skip_existing`

## 派生表格

- `seed1_signed_phiz_summary.csv`：seed1 新变体与基线的 ID/OOD/All PSNR 均值和差值。
- `seed1_scale_deltas.csv`：按尺度统计 `SC-INR+PhiZ - SC-INR`、`SC-INR-Signed - SC-INR`、`SC-INR - LTE`。
- `multiseed_core_per_seed.csv`：LIIF/LTE/SC-INR 三个核心模型每个 seed 的 ID/OOD/All PSNR。
- `multiseed_core_summary.csv`：LIIF/LTE/SC-INR 三个核心模型的 3 seed mean/std。

## 当前可支持的结论

- `SC-INR+PhiZ` 在 seed1 PSNR benchmark 上优于 `SC-INR`：ID `+0.067 dB`、OOD `+0.030 dB`、All `+0.042 dB`。这说明 feature-conditioned phase 是有希望的增量，但目前还只是 seed1 PSNR 证据。
- `SC-INR-Signed` 相对 `SC-INR` 在 seed1 上基本中性：ID `+0.005 dB`，OOD `-0.005 dB`，All `-0.002 dB`。signed omega 在参数化上更合理，但当前 PSNR 结果不足以支持“更强”的经验结论。
- 核心三模型 3 seed 汇总中，`SC-INR` 相对 `LTE` 的 OOD PSNR 提升稳定：mean `+0.0507 dB`，std `0.0047 dB`；All PSNR mean `+0.0325 dB`，std `0.0058 dB`。
- `SC-INR` 的 ID PSNR 不是当前优势：3 seed mean 相对 `LTE` 为 `-0.0040 dB`。论文表述应强调 decoder-side sampling consistency 对 OOD scale robustness 的帮助，而不是泛化成所有尺度/所有指标都更优。

## 尚不能支持的结论

- 不能据此宣称 strict scale equivariance。
- 不能把 `SC-INR+PhiZ` 作为最终主模型，除非继续补齐 consistency、texture RMSE、continuous-scale curve 和至少关键 seed 复现。
- 不能说 signed omega 已经经验优于原 SC-INR。更准确的说法是：signed omega 修复了非负 Cartesian omega 的方向表达风险，但 seed1 PSNR 未体现稳定收益。

## 主流程内模拟独立审查

- 实验协议审查：新增 benchmark 使用现有 `eval_full.py`、HR 现场下采样协议和 `epoch-best.pth`，未使用本地不可靠的预生成 LR。seed3 LTE 已补齐，核心多 seed 统计现在可用。
- 证据/过度宣称审查：`SC-INR+PhiZ` 的 PSNR 增益值得继续投入，但还缺少 consistency 和机制诊断；`SC-INR` 多 seed OOD 增益更稳，适合作为当前论文主线证据。
