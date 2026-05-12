# 实验：SC-INR-NoSinc

## 科学问题

在 signed feature-conditioned omega 和 feature-conditioned phase 之外，最终候选
`SC-INR` 是否确实受益于 analytic sinc observation response？

## 假设

如果 analytic response 是 decoder-side sampling consistency 的有效组成部分，去掉它应削弱
相对完整 `SC-INR` 的 OOD scale robustness 或 same-LR cross-scale consistency。

## 冻结协议

- 模型：`SC-INR`、`SC-INR-NoSinc`；评估时报告 `LTE`、`SC-INR-NoPhi`、
  `SC-INR-NoPhi-Signed` 作为对照。
- 数据划分：沿用 `configs/train-div2k/train-sc-inr.yaml` 的 DIV2K train/valid。
- 尺度：训练 `scale_max: 4`；benchmark ID x2/x3/x4，OOD x6/x8/x12/x16/x24/x30。
- Seeds：从 seed1 开始；不能把 seed1 结果写成 multi-seed 结论。
- checkpoint：`artifacts/checkpoints/seed1/sc-inr-nosinc/epoch-best.pth`。
- 指标：PSNR、PSNR-Y、SSIM-Y、texture/highpass RMSE、same-LR x8/x16/x30 -> x4 consistency。
- baseline：完整最终候选 `SC-INR`，保持 signed omega 和 `phi(z)`，仅 `use_sinc_response: true`。
- 禁止事后调整：看到 NoSinc 结果后不得调 omega bounds、phase 设置、数据协议或 checkpoint 选择。

## 命令

- smoke：
  `python -m py_compile src/models/sc_inr_adaptive.py`
- 训练：
  `CUDA_VISIBLE_DEVICES=<gpu> python entrypoints/train.py --config configs/train-div2k/train-sc-inr-nosinc.yaml --name sc-inr-nosinc --saveFolder artifacts/checkpoints/seed1 --seed 1`
- benchmark：
  `python entrypoints/eval_full.py --device <gpu> --save_root artifacts/checkpoints/seed1 --models SC-INR-NoSinc --output artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json`
- auxiliary：
  `python scripts/analysis/evaluate_seed1_aux_metrics.py --models LTE,SC-INR-NoPhi,SC-INR-NoPhi-Signed,SC-INR,SC-INR-NoSinc --datasets bsd100,urban100 --scales 4,8,16,30 --consistency_pairs '8,4;16,4;30,4' --max_images 10 --out artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1 --skip_visuals`

## 产物

- config：`configs/train-div2k/train-sc-inr-nosinc.yaml`
- checkpoint：`artifacts/checkpoints/seed1/sc-inr-nosinc/`
- raw result：`artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json`
- derived diagnostics：`artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/`
- log：`artifacts/checkpoints/seed1/sc-inr-nosinc/log.txt`

## benchmark 结果

seed1 benchmark 已完成。

- raw result：`artifacts/raw_results/seed1/benchmark_sc_inr_nosinc.json`
- derived summary：`artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-10/seed1_signed_phiz_nosinc_summary.csv`
- `SC-INR-NoSinc`：ID `30.9790`，OOD `22.7299`，ALL `25.4796`
- `SC-INR-NoSinc - LTE`：ID `-0.0659`，OOD `+0.0382`，ALL `+0.0035`
- `SC-INR-NoSinc - SC-INR`：ID `-0.1397`，OOD `-0.0467`，ALL `-0.0777`
- `SC-INR-NoSinc - SC-INR-NoPhi`：ID `-0.0724`，OOD `-0.0172`，ALL `-0.0356`

结论强度：`single-seed benchmark evidence plus seed1 auxiliary diagnostic`。

初步解释：移除 analytic sinc response 会削弱完整 `SC-INR`，尤其是 ID PSNR 和 ALL
平均。OOD 仍略高于 LTE，因此该 benchmark 支持 sinc 有贡献，但不能证明 sinc 是唯一原因。

## auxiliary 结果

seed1 auxiliary metrics 已完成。

- derived analysis：`artifacts/derived/diagnostics/sc_inr_nosinc_aux_metrics_seed1/`
- 协议：BSD100 / Urban100 前 10 张；x4/x8/x16/x30 quality；same-LR x8/x16/x30 -> x4 consistency。
- 覆盖：480 行 quality，300 行 consistency；共享模型 summary 与
  `sc_inr_final_aux_metrics_seed1` 完全一致。
- BSD100/Urban100 x8/x16/x30 OOD PSNR-Y 平均：
  `SC-INR-NoSinc` 21.7284 vs `SC-INR` 21.7218（`+0.0066 dB`）。
- OOD texture RMSE-Y 平均：
  `SC-INR-NoSinc` 0.152998 vs `SC-INR` 0.153208（`-0.000210`，越低越好）。
- Same-LR consistency PSNR-Y 平均：
  `SC-INR-NoSinc` 66.8555 vs `SC-INR` 49.6275（`+17.2280 dB`）。

解释：auxiliary consistency 没有显示 NoSinc 破坏 same-LR self-consistency，反而更高。
原因很可能是禁用 sinc 后 cell-dependent observation response 变为零，decoder 更接近
cell-independent point function。这说明 consistency metric 必须与 fidelity 和 response
diagnostics 联合解释。

## caveat

这个消融只能测试同训练协议下 removing sinc 是否削弱结果。它不能排除 optimization
variance、phase/omega compensation 等解释，也不能证明唯一因果。

## Response/Omega diagnostic

response/omega diagnostics 已完成。

- derived analysis：`artifacts/derived/diagnostics/response_omega_diagnostics_2026-05-10/`
- 协议：BSD100 / Urban100 前 5 张 response distribution；BSD100 / Urban100 前 5 张
  cell-only curve，同 x4 LR、同 coordinates，只改变 cell scale。
- `SC-INR-NoSinc` active attenuation mean 在 x4/x8/x16/x30 均为 `0`，因为模型实际使用 `W=1`。
- `SC-INR-NoSinc` cell-only RMSE-Y vs x4 在所有测试尺度均为 `0`。
- 完整 `SC-INR` active attenuation mean 非零：
  x4 `0.040495`，x8 `0.010570`，x16 `0.002672`，x30 `0.000762`。
- 完整 `SC-INR` cell-only RMSE-Y vs x4 在 x8/x16/x30 的平均为 `0.003255`，
  低于 LTE（`0.007943`），但高于 `SC-INR-NoSinc`（`0`）。

解释：NoSinc 的高 same-LR consistency 来自 zero active cell response，而不是更好的
footprint observation model。完整 `SC-INR` 保留有限 analytic cell response，同时避免 LTE
更大的 learned cell-phase sensitivity。
