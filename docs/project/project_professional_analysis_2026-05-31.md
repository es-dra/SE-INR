# SC-ASISR 项目专业分析（2026-05-31）

本文档是一次项目接手式审阅，目标是回答：项目现在研究什么、代码与证据支撑到哪里、哪些 claim 可以写、哪些风险会误导后续推进。

本次审阅未启动真实 subagent：当前运行时工具要求用户显式要求并行 agent 才能创建。以下包含主流程内模拟的独立审稿人视角，但它不是独立验证。

## 1. 项目定位

当前真正活跃仓库是 `SC-ASISR/`，外层 `/workspace/SE-INR` 是工作区层，包含数据集、本地配置和一个旧 `paper/` 目录。

项目主题是 arbitrary-scale image super-resolution（ASISR）中的 decoder-side scale sampling consistency。当前主方法 `SC-INR` 的核心不是严格数学意义的 scale equivariance，而是把输出 cell/footprint 从 LTE 的 learned phase path 中拿出来，改为 analytic sinc observation response。

最准确的当前研究问题是：

> LTE 允许 cell 通过 learned `h_p(c)` 直接改变 Fourier-like phase，这可能形成尺度外推时的纹理相位捷径。SC-INR 将内容参数 `coef(z) / omega(z) / phi(z)` 限定为 feature-conditioned，并让 cell 只通过 `W(omega,c)=sinc(omega_x c_x/2)sinc(omega_y c_y/2)` 调制 Fourier decoder branch 的观测响应。

推荐论文表述：

- decoder-side sampling consistency
- sampling-consistent Fourier implicit decoder
- scale-decoupled observation
- analytic footprint response

禁止升级表述：

- strict scale equivariance
- whole-network scale equivariance
- exact continuous box integral
- sinc 是唯一因果因素
- selected crops 证明平均视觉质量更好

## 2. 代码结构与入口

活跃代码结构清晰，已经从旧兼容路径收敛到 canonical 目录：

- `src/models/`：模型实现，核心是 `liif.py`、`lte.py`、`lte_noc.py`、`lte_phase_z.py`、`sc_inr_adaptive.py`、`sc_inr_eq.py`。
- `src/datasets/`：图像文件夹与 implicit SR wrappers。
- `entrypoints/train.py`：训练入口。
- `entrypoints/eval_full.py`：正式离散尺度 benchmark 入口。
- `scripts/analysis/build_canonical_benchmarks.py`：当前论文 benchmark 表的 canonical 生成入口。
- `configs/registry/models.yaml`：模型展示名、checkpoint、raw key 与旧名映射。
- `configs/registry/protocols.yaml`：评估协议入口。
- `artifacts/derived/benchmarks/`：当前论文主 benchmark 入口。
- `paper/claims_evidence_matrix.md`：claim 证据边界。
- `paper/ARTIFACTS_ALLOWED.md`：论文可引用 artifact 白名单。
- `docs/project/current_state.md`：当前状态恢复入口。

根目录 symlink 仍保留旧兼容路径，例如 `models -> src/models`、`save -> artifacts/checkpoints/seed1`、`results -> artifacts/results`。这对旧命令恢复有价值，但新命令和论文应继续使用 canonical 路径。

## 3. 方法实现核对

### LIIF

`src/models/liif.py` 实现 local implicit image function：

- 取局部 feature `q_feat`；
- 构造相对坐标 `coord - q_coord`；
- 可选把 `cell` 拼入 MLP 输入；
- local ensemble 通过四邻域面积权重融合。

因此不能写“LIIF 完全不使用 cell”。更准确说法是：LIIF 可以把 cell 当普通条件变量交给 MLP，但没有显式的 footprint/frequency response 结构。

### LTE

`src/models/lte.py` 中：

- `coef(z)` 和 `freq(z)` 由 feature map 卷积预测；
- `phase = nn.Linear(2, hidden_dim//2, bias=False)` 接收 `rel_cell`；
- basis 为 `cos(pi*(omega(z)*delta + h_p(cell)))` 和 `sin(...)`。

这与项目叙事一致：LTE 的 cell-conditioned learned phase 是本项目主要质疑的尺度路径。

### LTE 诊断变体

`lte_noc.py` 移除 `h_p(c)`，phase 固定为 0。

`lte_phase_z.py` 把 phase 改为 `phase_conv(z)`，即 feature-conditioned phase，不接收 cell。

这两个变体能隔离“删除 cell phase”和“保留 feature phase”的影响。

### SC-INR

`src/models/sc_inr_adaptive.py` 是核心实现：

- `SCINRAdaptive`：旧 no-phase 主线 `SC-INR-NoPhi`，`phi=0`。
- `SCINRAdaptiveSigned`：signed bounded omega，无 phase。
- `SCINRSignedPhiZ`：当前 final `SC-INR`，signed omega + feature-conditioned `phi(z)`。
- `SCINRNoSinc`：final 架构去掉 analytic sinc response，`W=1`。

关键路径：

- `omega_map = omega_bound * tanh(raw_omega)`；
- `phase_map = phase_conv(feat)`，只依赖 feature；
- `q_phase = sum(q_omega * rel_coord) + phi(z)`；
- `W = sinc(omega_x * c_x / 2) * sinc(omega_y * c_y / 2)`；
- `fourier_feats = [cos(pi*q_phase)*W, sin(pi*q_phase)*W]`；
- `inp_imnet = q_coef * fourier_feats`。

这个实现支持“cell 不进入 learned phase、只进入 analytic response”这一主张。但更严谨地说，该结论限定在 Fourier decoder branch；模型整体仍包含 local ensemble 面积融合和 `upinput` 的 bilinear residual path，不能说整个前向图是精确解析 box average。

## 4. 数据与评估协议

训练配置采用 DIV2K HR online bicubic downsampling：

- `inp_size: 48`
- `scale_max: 4`
- `sample_q: 2304`
- `batch_size: 16`
- `epoch_max: 1000`

评估协议 `benchmark_discrete`：

- 数据集：Set5、Set14、BSD100、Urban100；
- ID scales：x2/x3/x4；
- OOD scales：x6/x8/x12/x16/x24/x30；
- 指标：benchmark Y-channel PSNR，边界 shave = scale；
- checkpoint 规则：`epoch-best.pth`。

`entrypoints/eval_full.py` 默认从 `SEINR_DATA_ROOT` 或外层 `../Data` 读取 HR，并现场 downsample。这个选择比引用本地预生成 LR benchmark 更干净，也避免历史 LR/HR 不一致问题。

## 5. 当前证据状态

### 主 benchmark

`python scripts/analysis/build_canonical_benchmarks.py --out /tmp/sc_asisr_bench_check` 可正常重建 benchmark 表，且与 `artifacts/derived/benchmarks/` 完全一致。

当前主表数据：

- `SC-INR`：ID `31.0738 ± 0.0582`，OOD `22.7578 ± 0.0316`，ALL `25.5298 ± 0.0403`。
- vs LIIF：ID `+0.0437 ± 0.0757`，OOD `+0.0328 ± 0.0373`，ALL `+0.0364 ± 0.0499`。
- vs LTE：ID `-0.0048 ± 0.0836`，OOD `+0.0504 ± 0.0417`，ALL `+0.0320 ± 0.0555`。

专业判断：

- OOD/ALL 平均为正，这是当前最稳的主结果。
- 增益是小幅而非大幅；n=3，std 与 delta 同量级，不能写成强统计结论。
- ID vs LTE 轻微为负，反而有利于把 claim 收缩为“尺度外推/OOD 方向的 sampling consistency”，而不是泛化成所有尺度均优。

### SC-INR-NoPhi

旧 no-phase 变体有 3-seed core benchmark：

- vs LTE：ID `-0.0040 dB`，OOD `+0.0507 dB`，ALL `+0.0325 dB`。

它是更早、更稳的 decoder-side sampling consistency 证据，但不应继续被误写成当前 final `SC-INR`。

### SC-INR vs SC-INR-NoPhi

当前 final `SC-INR` 相对 `SC-INR-NoPhi` 的结构增量只适合写 seed1 context：

- OOD `+0.0295`
- ALL `+0.0421`

不能写成 feature-conditioned phase 在 3 seed 下稳定优于 no-phase。

### NoSinc

`SC-INR-NoSinc` seed1：

- vs full `SC-INR`：ID `-0.1397`，OOD `-0.0467`，ALL `-0.0777`。

这支持 analytic sinc response 对 final seed1 fidelity 有贡献，但不能证明唯一因果。更关键的是，NoSinc 的 same-LR consistency 反而更高，说明该 consistency 指标会奖励 cell-insensitive decoder，不能单独作为 sampling correctness 证据。

### Footprint/机制诊断

footprint oracle、effective amplitude、LTE-vs-SC-INR unified mechanism gate 都是 seed1 小样本 diagnostic。它们共同支持以下较弱但有价值的机制解释：

- LTE 有可观测 learned `h_p(c)` cell-phase 信号；
- SC-INR 有非零 analytic response/effective-amplitude proxy；
- SC-INR 在 HR box-average proxy 上比 LTE/NoSinc 略好地跟随 footprint target；
- 但这些不替代全量 benchmark，也不证明 exact continuous integral。

### 定性图与局部优势区间

用户确认 Urban100 x8 selected examples 可以展示局部结构更清晰。advantage-region 诊断补充说明 selected crop 来自预注册候选池，而不是手挑无上下文孤例。

但它仍是 seed1/局部/candidate 层级，不能写成平均视觉质量改善。

## 6. 论文与文档状态

活跃论文草稿在 `SC-ASISR/paper/draft/`。该草稿总体与当前证据对齐：

- 没有把方法写成 strict scale equivariance；
- 把 final `SC-INR` 对 `LIIF/LTE` 的 3-seed 结果作为主表；
- 把 `SC-INR` vs `SC-INR-NoPhi` 限定为 seed1 context；
- 对 same-LR consistency 和 NoSinc caveat 写得谨慎。

`docs/project/model_details_liif_lte_scinr.md` 最近被压缩成方法脉络说明，质量比长篇混合笔记更适合作为后续恢复入口。它把普通 INR、LIIF、LTE、SC-INR 的变量组织讲清楚，并明确禁止把 decoder 中间 Fourier 参数写成最终 RGB 物理频谱。

需要注意的结构风险：

- 外层 `/workspace/SE-INR/paper/` 仍像一个可编译论文工作区，且包含旧 `SC-INR-Adaptive` / no-phase / 旧路径叙事。
- 虽然外层 `paper/README.md` 提醒旧 strict scale-equivariant SE-INR 草稿已清空且不要复用，但其 `sections/*.tex` 仍有过时内容。
- 后续论文写作应只使用 `SC-ASISR/paper/draft/`，或把外层 `paper/` 明确退役为 README-only/archive，避免后续 agent 或用户误入旧入口。

## 7. 工程健康

强项：

- 模型 registry 已经明确解决 raw key 语义漂移问题。
- `build_canonical_benchmarks.py` 把 raw key、source file、canonical model、seed、protocol 显式展开，避免后续从 JSON key 猜语义。
- artifacts 分层清晰：raw、derived benchmark、diagnostics、paper figures、legacy 各自有边界。
- `.gitignore` 已限制 checkpoint、logs、大图、诊断图进入源码管理。
- 新增文档和 README 大多为中文，适合用户审阅。

弱项：

- 没有正式 pytest/property test。当前验证主要靠 `py_compile`、benchmark 重建和人工检查。
- `src/models/B_Conv.py` 仍有两个 TODO，属于 Rot-E/EQ 路线的潜在风险。
- `entrypoints/test.py` 与 `eval_full.py` 有重复评估逻辑，后续若改 metric/shave 规则可能分叉。
- `train.py` 默认 `.cuda()` 较多，CPU/多设备通用性弱；这对当前 GPU 训练不是阻塞，但对可复现环境说明有影响。
- `torch.meshgrid` 在 `utils.make_coord` 未显式 `indexing`，未来 PyTorch 版本可能有 warning 或行为审计问题。

## 8. 最强反对意见

审稿人最可能攻击的点不是“有没有新结构”，而是“证据是否足以说明这个结构带来了稳定科学收益”。

最强反对意见：

1. 主增益很小，3 seed std 不低，必须避免写成强优势或 SOTA。
2. sinc response 的因果性证据仍主要是 seed1 消融 + proxy 机制诊断；NoSinc 只跑单 seed。
3. same-LR consistency 已被 NoSinc 负控击穿，不能作为独立主证据。
4. final `SC-INR` 相对 `SC-INR-NoPhi` 的稳定性没有 3-seed 直接支持，feature phase 的贡献应保持 seed1 context。
5. 机制诊断中的 `h_p(c)` delta、active response delta、oracle RMSE 是不同物理量，不能把数值大小直接等价比较。
6. 若论文仍保留外层旧 paper 或旧名 `SC-INR-Adaptive`，会让读者误解当前 final method 与证据链。

## 9. 下一步建议

优先级从高到低：

1. 论文入口清理：明确退役外层 `/workspace/SE-INR/paper/`，或只保留 README 指向 `SC-ASISR/paper/draft/`。
2. 论文主表落地：从 `artifacts/derived/benchmarks/` 生成最终 LaTeX 表，禁止手抄 raw JSON。
3. final `SC-INR` auxiliary 补强：如果要在正文强调 final variant 的机制，至少补 final candidate 的更系统 auxiliary/footprint 诊断；否则将机制 claim 写成 decoder family 证据。
4. NoSinc 扩 seed：如果想把 sinc response 作为强消融 claim，优先补 seed2/3 或明确它只是 seed1 diagnostic。
5. 添加最小 property tests：检查 SC-INR 的 `phase_conv` 不接收 cell、NoSinc 的 `W=1`、同 coord 不同 cell 下 full SC-INR 与 NoSinc 的响应差异、输出 shape/device/dtype/finite。
6. 收缩旧入口：减少旧 `SC-INR-Adaptive`、`SC-INR+PhiZ`、外层旧 paper、legacy figures 对活跃文档的可见度。

## 10. 本轮结论

这是一个已经从“想法探索”进入“证据边界治理”阶段的研究项目。当前代码和主要文档基本自洽：方法确实实现了从 LTE learned cell phase 到 SC-INR analytic footprint response 的 decoder-side 约束转移；benchmark 也支持小幅 OOD/ALL 正增益。

但项目还不能被包装成强性能论文或严格理论论文。它更适合写成：

> 一个对 LTE 类 Fourier implicit decoder 的尺度路径重构：把 cell 从 learned phase shortcut 改为 analytic footprint response，并在当前 ASISR protocol 下带来小幅 OOD/ALL PSNR 改善，同时通过诊断说明该机制与 cell-insensitive 负控、LTE learned phase path 的差异。

如果继续推进论文，下一阶段的质量关键不是再堆更多图，而是把 final method、主表、消融、机制诊断和旧入口彻底对齐，防止旧命名和单 seed 诊断把 claim 带偏。
