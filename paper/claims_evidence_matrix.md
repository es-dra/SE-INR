# 当前分析论文的贡献与证据

| 主张 | 原件 | 边界 |
| --- | --- | --- |
| cell平均收益不排除个体失败；过冲与方向不对需区分 | [自然图及非零参照](../experiments/october2026/补证-20261018/cell-reference-v2/)、[完整曲线/稳健性](../experiments/october2026/补证-20261018/readout-closure-v1/) | 固定cohort和参照；分型使用真值，非新定理或部署预测器。 |
| 固定ReLU开关不能救回多数焦点方向失败 | [解码对照](../experiments/october2026/补证-20261018/decoder-path-v1/) | 固定映射仍含解码权重，不代表特征独立出错。 |
| 换回旧权重仅救回1/16，原漂移解释未成立 | [第一阶段](../experiments/october2026/补证-20261018/sinc-path-patch-v1/) | 不隐去负结果；不把后续分配解释冒称原先预言。 |
| 原生等总RMS改变量的分配干预救回16/16焦点中频，全32由31负/1正到32正；低频中心RGB16正到16负 | [确认阶段](../experiments/october2026/补证-20261018/sinc-allocation-confirm-v1/) | 单SC权重/×4/已见受控样本；非逐点/RGB等范数、真实频率估计错误或通用修复。 |

更早的失败预测、校准、G1/L2/G2及谱歧义探索见[完整来源](current/SOURCES.md)。下方保留服务器旧方法证据矩阵原文；其中冻结/晋级约束针对旧新方法身份。

---

# Claim 与证据矩阵

本文档约束冻结内部技术记录中的 claim 强度，避免表述超过已有证据。

## 2026-08-20 Q0 与 acquisition evidence 价值 gate

`artifacts/derived/diagnostics/spatial_source_footprint_q0_2026-08-20/q0_triage.json` 只证明冻结的 PCIE/PART、
RealSR 与 DPDD 三个公开家族没有一个原子 release 同时提供可下载数据、空间 operator mapping、明确数据许可、
paired target、fixed split 与官方 PSF-aware baseline。裁决是 `blocked-by-public-calibrated-data`，不是
spatially varying operator prior 的科学 negative；禁止用 synthetic DIV2K blur 追溯冒充现实数据，也禁止
“首个 PSF-aware restoration”claim。

`artifacts/derived/diagnostics/acquisition_robustness_existing_evidence_value_audit_2026-08-20/value_audit.json` 将
既有证据裁决为 `internal-case-study-only`。Gaussian epoch-100 三模型三 seed 矩阵只允许说明固定 bicubic
baseline 对该 synthetic source blur 共同脆弱、未观察到预注册 retention 优势；PC0 只允许作同一
Gaussian-LIIF source 的 classical matched adjacent control。A0 被简单坐标 control 吸收，两个 token pilot
只能各自内部解释。禁止称现实 acquisition robustness、统一 benchmark、paper-ready negative result、SC-INR
普遍更鲁棒/更不鲁棒、`kappa`/phase/sinc/cell 因果、尺度等变或新方法贡献。

## 2026-08-07 P1.12-C SC-INR 路线收口

P1.12 双 baseline 来源与公平性硬门失败后，SC-INR 新方法和投稿 claim 已停止。现有 benchmark、
checkpoint、raw result、公式与诊断只保留为内部可复现事实。外部 provenance 改变最多允许重开
P1.12 审计，不会自动恢复 novelty；直接 baseline、LTE factorial、有效 observation controls 和超过
seed/协议波动的结果仍是后续独立门槛。完整判决见
`docs/project/sc_inr_route_closeout_2026-08-07.md`。

## 2026-08-05 prior-art gate

IPE-LIIF（arXiv:2112.05756）已经在 ASISR 中推导 pixel-area integrated Fourier
encoding 的 sinc response；Thera（arXiv:2311.17643，TMLR 2025）进一步实现内容条件
局部正弦场与解析 Gaussian PSF。因此以下 claim 立即标为 `blocked-for-claim`：

- SC-INR 首次把 pixel footprint 或 analytic sinc 引入 arbitrary-scale SR；
- 当前 pre-MLP sinc 构成 exact box-integrated RGB decoder；
- 现有结果已经建立新的通用 sampling-consistency principle；
- 当前方法论文 novelty 已成立。

下表中的 benchmark 数值只能作为现有实现事实，不能单独恢复方法论文 claim。即使未来来源条件
改变，也必须重新立项并逐项通过公平性、归因和指标有效性门槛。详细 prior-art gate 见
`docs/project/literature_novelty_audit_2026-08-05.md`。

## 2026-08-06 P1.7 identifiability gate

`artifacts/derived/deterministic_detail_no_go_p17/` 只允许一个窄 claim：在冻结 finite-spline
alias pair 上，不存在稳定、确定性、source-only 且对两支都逐样本正确的 nullspace-detail
selector。它不允许写成“自然图像高频不可预测”，不验证 conditional distribution，不证明
learned scale-equivariant INR，也不解锁 P2。zero detail 是该 balanced ambiguity 的 MSE 最优
条件均值，但不是完整 SR 输出或真实 branch recovery。

## 2026-08-06 P1.8 conditional-mean applicability gate

`artifacts/derived/conditional_mean_intertwining_p18/` 允许两个窄 claim：joint pushforward 一般只
推出 transformed-detail 的 fiber average；强 conditional-mean intertwining 还需要 bounded
linearity 与 `rho m_s(Y_s)` 对 `UY_s` 可测。P1.6 的正控只在 26 维 classical restricted range
成立；P1.7 的完整 physical target 不满足 deterministic joint-closure 前提。禁止写成 natural-image
distribution closure 已验证、physical balanced conditional mean 已被反证、概率模型已验证、learned
scale-equivariant INR 已成立或 P2 已解锁。

## 2026-08-06 P1.9 probabilistic transport gate

`docs/project/probabilistic_scale_transport_p19_route_audit_2026-08-06.md` 允许三个窄判断：简单
`K_t(Uy)=rho#K_s(y)` 在 residual 会泄漏进 target observation 时不合法；section-dependent coupled
`(y,r)` transport 是可定义的 joint-law 语义；当前单观测 ASISR 数据和指标不能独立验证真实
conditional coupling/calibration，因此概率 P2 仍冻结。禁止写成 diffusion/概率 SR 已被否定、
coupled transport 已形成新方法、P0--P1.9 已足以投稿，或 multi-observation 路线已验证。

## 2026-08-06 P1.10 multi-observation observability gate

`artifacts/derived/multi_observation_observability_p110/` 只允许一个窄负结果：在冻结 finite-spline、
box-sensor 与五个预注册 target-independent 物理 bundle 下，增加观测 rank 没有稳定覆盖三个
held-out target row spaces；half-pixel shifts 是综合证据最强但仍失败的候选。该结果只作 research
gate，不进入当前 SC-INR 主论文。禁止写成新 generalized-sampling theorem、多帧/多观测 SR 无效、
自然图像近似恢复不可能、learned fusion 已被否定、尺度等变 INR 已做成，或 P2/GPU 已解锁。

## 2026-08-07 P1.11 route-reset gate

`docs/project/scale_equivariant_inr_p111_route_reset_2026-08-07.md` 是研究路线决策，不是论文主证据。
它只允许写：当前没有长期尺度等变 INR 突破路线通过硬门；多观测路线等待外部 acquisition，单图
统计路线在对象定义前阻塞，SC-INR 只获准做封顶成本的 direct-prior evidence closeout。禁止把
`none_can_advance` 写成所有尺度等变研究被否定，或把 C 的可执行性写成 SC-INR novelty 已恢复。

## 2026-08-07 P1.12 direct-prior baseline gate

`docs/project/scale_equivariant_inr_p112_baseline_compatibility_audit_2026-08-07.md` 只允许写：Thera 有
固定 commit 和 Apache-2.0 官方实现，但 model-native recipe 与 SC-INR 不是严格同预算消融；
IPE-LIIF 论文核心区域积分公式可独立实现，却没有可确认的官方代码、commit、软件许可证或
checkpoint。预冻结双 baseline 条件未满足，因此不进入 P1.13，SC-INR 新方法 claim 保持停止。
禁止把“论文公式可实现”写成“官方 IPE-LIIF 已复现”，把 Thera 的作者表格与本地结果拼接，或把
当前负判决写成 IPE/Thera 性能优劣已经得到实验验证。

| Claim | 证据 | 强度 | 允许表述 | 禁止过度表述 |
| --- | --- | --- | --- | --- |
| `SC-INR-NoPhi` 相对 LTE 改善 OOD scale robustness。 | active benchmark 入口是 `artifacts/derived/benchmarks/benchmark_all_models_long.csv`；阶段 provenance 在 `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-09/multiseed_core_summary.csv`，OOD 相对 LTE 为 `+0.0507 dB`。 | 当前协议/core models 下已验证。 | `SC-INR-NoPhi` 在当前 benchmark protocol 下相对 LTE 有稳定小幅 OOD PSNR 改善。 | `SC-INR` 严格 scale-equivariant；大幅 SOTA gain。 |
| `SC-INR-NoPhi` 改善 same-LR empirical cross-grid agreement。 | `artifacts/derived/diagnostics/seed1_aux_metrics_all8/paper/consistency_table.tex`：BSD100 相对 LTE `+8.61 dB`，Urban100 相对 LTE `+7.77 dB`。 | seed1 diagnostic；不是 box-observation 机制证据。 | no-phase variant 在当前 bicubic cross-grid diagnostic 下与直接低倍率输出更一致。 | decoder-side sampling consistency 已证明；whole network scale equivariant。 |
| 冻结实现 `SC-INR` 在当前 3 seed protocol 下相对 LIIF/LTE 的 OOD/ALL PSNR mean delta 为正。 | `artifacts/derived/benchmarks/paper_main_3seed_summary.csv`：`SC-INR` OOD `22.7578 ± 0.0316`，ALL `25.5298 ± 0.0403`；`paper_main_3seed_paired_delta.csv`：相对 LIIF 的 OOD/ALL 为 `+0.0328 ± 0.0373` / `+0.0364 ± 0.0499`，相对 LTE 的 OOD/ALL 为 `+0.0504 ± 0.0417` / `+0.0320 ± 0.0555`。seed1 context 在 `paper_context_seed1.csv`：`SC-INR` 相对 `SC-INR-NoPhi` 的 OOD/ALL 为 `+0.0295` / `+0.0421`。 | 当前协议下的 3 seed 实现事实；增益与 seed 波动同量级，不能升级为稳定方法优势。 | 在当前三 seed bicubic 协议下，观察到 SC-INR 相对 LIIF/LTE 的正 OOD/ALL mean delta。 | feature phase 是稳定 multi-seed gain；SC-INR 稳定改善 scale extrapolation；`SC-INR` 是 consistency 最强变体或严格 scale-equivariant。 |
| first-10 逐图统计支持 SC-INR 的 OOD 增益不是单个尺度/图像偶然撑起。 | `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/paired_delta_bootstrap_ci.csv`：Set5 全 5 张、Set14/BSD100/Urban100 sorted 前 10 张，3 seeds、9 scales。相对 LTE 的 OOD mean/median/win-rate 为 `+0.0434` / `+0.0320` / `0.746`，cluster bootstrap mean CI `[+0.0326,+0.0537]`；相对 LIIF 的 OOD mean/median/win-rate 为 `+0.0291` / `+0.0216` / `0.689`，CI `[+0.0200,+0.0389]`。 | 中等预算 first-10 diagnostic；不替代全量 benchmark。 | 在 sorted first-10 子集上，SC-INR 的 OOD paired deltas 对 LIIF/LTE 呈正 median 和高于 50% 的 win-rate，补充说明平均增益不是单个样例独立支撑。 | 这是全量逐图统计；证明所有数据集/所有图稳定获胜；可以替代 `paper_main_3seed_summary.csv`。 |
| 预注册局部 crop 候选池显示 SC-INR 在 OOD 局部区域相对 LTE/LIIF 有正 median delta，但优势较温和。 | `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_per_crop.csv` 和 `external_advantage_summary_by_scale.csv`：BSD100/Urban100 sorted 前 30 张、seed1、scales x4/x8/x16/x30、固定 96 crop/stride 96、每图每尺度最多 32 crop。x8/x16/x30 对 LIIF 和 LTE 的 pairwise median 均为正；同一 crop 同时赢 LIIF/LTE 的比例约 `0.417-0.457`。底层 `local_crop_metrics.csv` 和 `per_image_per_scale_metrics.csv` 只作本地可再生成复现表，不作为源码管理入口。 | seed1 局部 diagnostic；候选池预注册但不是 multi-seed。 | 预注册局部候选池支持：SC-INR 的局部 OOD 外部基线优势存在但温和；正文 selected crops 必须和完整候选池统计一起解释。 | selected crops 证明平均视觉质量更好；局部优势强且普遍；NoSinc/NoPhi 已被局部指标充分击败。 |
| SC-INR 相对 LIIF/LTE 的明显可视优势主要出现在 selected OOD 结构 crop，而不是全局普遍区域。 | `artifacts/derived/diagnostics/sc_inr_advantage_midbudget10_2026-05-21/external_advantage_region_pool.csv` 是候选池入口；本地 `figures/external_advantage_crops/` 和 `figures/external_advantage_context_zoom/` 是渲染候选图目录，论文采用前需迁移到 `artifacts/derived/paper_figures/` 并登记。候选只从 OOD、非平坦结构 crop 中选，排序指标为 `min(SC-INR-LIIF, SC-INR-LTE)`。top examples 包括 Urban100 `img_011.png` x8 crop12，min external Δ `+4.663 dB`；Urban100 `img_011.png` x16 crop18，`+4.226 dB`；BSD100 `167062.png` x30 crop12，`+3.999 dB`。x4 sanity pool 的 min external Δ 约为 0，只能作为低倍率边界对照。 | seed1 selected qualitative candidates，带 top/neutral/failure pool；主用于定性展示。 | 可以写：selected examples illustrate the external-baseline advantage regions observed in the pre-registered crop pool，尤其是 OOD 建筑/纹理/高频结构 crop；x4 作为 sanity/boundary 展示低倍率差异很小。 | “这些图证明平均视觉质量更好”；“所有高频区域都明显优于 LIIF/LTE”；“x4 也有明显局部优势”；“这是消融因果证明”。 |
| feature-conditioned phase 改善 selected examples 的可见局部结构。 | 用户确认并重导出的图在 `artifacts/derived/paper_figures/qualitative_selected_seed1/`：Urban100 img004 x8 中 `SC-INR` crop PSNR-Y `21.4880`，LTE `19.0226`，`SC-INR-NoPhi` `20.8467`；img012 x8 中 `SC-INR` `23.2229`，LTE `20.4458`，`SC-INR-NoPhi` `19.6163`。 | 定性候选证据。 | selected qualitative examples 显示更清晰的局部结构。 | 未给分布证据时宣称 average visual quality 更好。 |
| analytic sinc response 对最终候选有贡献。 | `SC-INR-NoSinc` 出现在 active seed1 context table `artifacts/derived/benchmarks/paper_context_seed1.csv`；provenance 在 `artifacts/legacy/derived_analysis/benchmark_provenance/benchmark_progress_2026-05-10/seed1_signed_phiz_nosinc_summary.csv`：相对完整 `SC-INR` 的 ID/OOD/ALL 为 `-0.1397 dB` / `-0.0467 dB` / `-0.0777 dB`。当前文件树中没有旧 `sc_inr_nosinc_aux_metrics_seed1/`，不能引用其 consistency 数值。 | seed1 benchmark/fidelity 证据；机制因果仍需诊断配合。 | 移除 analytic response 会削弱完整 seed1 benchmark，说明 sinc observation term 可能贡献最终候选 PSNR。 | gain 只由 sinc 导致；sinc 已被证明 necessary；当前指标下 NoSinc 破坏 sampling consistency；引用缺失的 old auxiliary 目录。 |
| same-LR empirical cross-grid agreement 是 diagnostic，不是 observation modeling 正确性的独立证明。 | 当前入口是 `artifacts/derived/diagnostics/seed1_aux_metrics_all8/paper/consistency_table.tex`；实现使用 bicubic resize 比较不同输出网格。NoSinc caveat 的旧目录当前缺失，需重跑后才能作为数值证据。 | 原理性 caveat + 部分 seed1 diagnostic；NoSinc 数值需重跑。 | 该指标只描述当前 bicubic cross-grid agreement，必须与 fidelity、声明的 observation operator 和 negative controls 联合解释。 | 更高数值等于更正确的 sampling model；引用缺失的 NoSinc consistency 数值。 |
| `SC-INR` 的 cell 不进入 learned phase/frequency，但仍通过显式 response 影响 decoder 输入。 | `artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_intervention/cell_intervention_metrics.csv`：固定 x4 LR/query，只改变 cell multiplier。selected examples 中 `SC-INR` cell sensitivity 非零，`SC-INR-NoSinc` 和 `LTE-PhaseZ` 基本为 0；top selected case 中 `SC-INR` tracking RMSE `0.0423`，低于 LTE `0.0455` 和 NoSinc `0.0709`。 | 小样本 mechanism diagnostic。 | `SC-INR` 保留非零 cell response，同时避免 LTE 的 learned cell-phase path；该结论只说明机制行为，不替代 benchmark。 | `SC-INR` scale-decoupled 或严格 scale-equivariant；`SC-INR` 忽略 scale/cell；selected diagnostic 证明平均视觉质量更好。 |
| cell 在 `SC-INR` 中调制 feature-level effective amplitude，而不是 learned phase。 | `artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_phase_proxy/cell_phase_proxy.csv` 显示 `LTE` 有 `cell -> phase`，x30 相对 x4 mean phase delta `0.1747`，而 `LTE-PhaseZ`、`SC-INR`、`SC-INR-NoSinc` 为 0。`effective_amplitude/effective_amplitude_summary.csv` 显示 `SC-INR` 高频 active `|W|` 从 x2 `0.6356` 增至 x30 `0.9979`，energy ratio 从 `0.3739` 增至 `0.9953`；`SC-INR-NoSinc` active response 恒为 1。 | 当前实际存在的小样本 feature-level 机制诊断。 | 在送入 MLP 前，SC-INR 的 coefficient-pair magnitude proxy 被 analytic `W(omega,c)` 调制，且 cell 不直接进入 phase；NoSinc 负控没有 active response。 | 最终 RGB 频谱振幅严格等于 `A(z)W(omega,c)`；sinc 是唯一因果；该诊断替代 benchmark、数据集级趋势或多 seed。 |
| `SC-INR` 的 cell path 在 selected examples 中更接近 footprint observation，但不是全局胜过所有 baseline。 | `artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_intervention/README_zh.md` 和 `cell_intervention_metrics.csv`：top selected case 中 `SC-INR` tracking RMSE `0.0423`，低于 LTE `0.0455` 和 NoSinc `0.0709`；18 个候选均值中 `SC-INR` 没有稳定超过 LIIF。 | 小样本 selected mechanism demonstration。 | 可写 selected examples 支持 `SC-INR` cell-induced change 区别于 LTE/NoSinc，并更接近 HR box-average oracle；必须说明小样本且不是 benchmark。 | `SC-INR` 全局视觉质量更好；真实连续图像 exact box integral 已证明；sinc 唯一因果；该诊断替代 3-seed benchmark。 |
| 旧 footprint/oracle 与 LTE-SC-INR 统一机制诊断需要重跑后才能引用。 | 旧路径 `artifacts/derived/diagnostics/footprint_oracle_2026-05-15/`、`artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/`、`artifacts/derived/diagnostics/response_omega_diagnostics_2026-05-10/` 当前不存在。 | missing/rerun_required。 | 这些结论可作为历史线索；进入论文前必须重跑、检查、登记 artifact。 | 把不存在路径继续写成 cite_ok；用旧数值支撑当前正文 claim。 |
| `SC-INR-EQ` 是可行的 exploratory Rot-E integration，不是新主方法。 | `SC-INR-EQ` 只在 seed1 context table `artifacts/derived/benchmarks/paper_context_seed1.csv` 和 raw provenance `artifacts/raw_results/seed1/benchmark_sc_inr_eq.json` 中出现；36/36 entries finite，ID/OOD/ALL 为 `31.1513` / `22.7779` / `25.5691`。相对 `LTE-EQ` 的 ID/OOD/ALL 为 `+0.0841` / `+0.0749` / `+0.0780`；相对最终 `SC-INR` 为 `+0.0326` / `+0.0014` / `+0.0118`。旧 active mechanism-suite mirror 当前缺失；完整历史产物在 `artifacts/legacy/derived_analysis/mechanism_suite_2026-05-12/`。 | 初步 single-seed exploratory evidence；Rot-E property tests 仍缺失。 | `SC-INR-EQ` 显示 SC-INR decoder contract 可以与 Rot-E plumbing 结合，在 single-seed benchmark 上相对 `LTE-EQ` 有小幅增益，并与最终 `SC-INR` 基本打平。 | `SC-INR-EQ` 是新主方法；Rot-E integration 证明 strict scale equivariance；single-seed `SC-INR-EQ` 是 paper-level final claim。 |
