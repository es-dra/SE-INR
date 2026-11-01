# 仓库审计与方法方向评估（Claude，2026-11-01）

范围：通读全部 300 个跟踪文件（代码、配置、18 个阶段的研究记录、论文、评审）。本轮不改代码、不跑实验；给出清理方案和在正式推进实验前的方向判断。服务器结果（试点 `claude-20261029-sp-pilot-v1`）尚未回填，所有判断不依赖它。

## 一、仓库审计

总体：仓库结构清楚，证据链完整，公开快照机制合理。问题不在"垃圾多"，而在三类：从原始 LIIF 代码库继承的死代码、研究阶段之间的复制粘贴、以及少量过时叙述。按风险分三档。

### A 档：可直接清理（不影响任何冻结记录或 checkpoint 加载）

| 位置 | 问题 | 处理 |
| --- | --- | --- |
| `entrypoints/train.py` | `lambda_cons`/`return_eq`/`cons_ratio` 一致性损失分支：没有任何模型实现 `return_eq`，开启即崩溃；`show_tempImage`/`Im_dataset` 中间图导出；`warmup_cosine_lr`/`cosine_lr` 无配置使用 | 删除三条分支，训练脚本可缩约 1/3 |
| `entrypoints/test.py` | `args.save_path` 导图分支、注释掉的 `--window` 参数 | 删除 |
| `src/models/liif.py` | `ImfunctionObserve`、`coordGen` 旧可视化方法，无调用 | 删除 |
| `src/utils.py` | `calc_ssim`（依赖未声明的 `pytorch_ssim`）、`save_results*`、`save_psnr`、`create_gaussian_kernel`（中文注释）、`normalized`、`setRange`：全部无调用 | 删除；`calc_psnr` 保留 |
| `src/datasets/wrappers.py` | 5 个 wrapper 只有 `sr-implicit-downsampled` 被任何配置引用；`sr-implicit-paired`/`noGT`/`discret-downsampled`/`uniform-varied` 为上游遗留 | 删除 4 个，文件由 360 行缩到约 90 行 |
| `studies/two-coordinates/12-gauss-degradation/gauss_wrapper.py` | 与 `src/datasets/gauss_wrapper.py` 逐字节相同 | 删除副本，README 指向 `src/` |
| `studies/two-coordinates/16-residual-structure/localization-v2/localize-source.py` | 与同目录上级 `localize.py` 逐字节相同 | 删除副本 |
| `paper/refs.bib` | `kawai2026frequency`、`pak2023btc`、`wang2026scaleanchoring` 未被引用 | 删除或在正文引用 |
| `README.md`、`studies/two-coordinates/README.md` 顶部状态行 | 仍写 "Study status 2026-10-23"，没有提到方法试点 | 更新 |

### B 档：建议整理，但要先确认

1. **`studies/two-coordinates/01–05` 的 `protocol.json` 是五份相同文件**（都是阶段 01 的协议），各阶段真正的协议是 `protocol_vN.json`。这会误导读者。建议删除 02–05 的同名副本，README 已经链接 `protocol_vN.json`。需确认 `FROZEN.sha256` 是否把它们计入；若计入，保留文件但在 README 注明。
2. **`run_v3.py` 在 03/04/05 三份相同**，`score_v5.py`/`score_v6.py`/`score_v8.py` 等为逐阶段复制改版。这是"冻结当时脚本"的设计选择，公开快照里保留是合理的；不建议合并，但建议在 `studies/two-coordinates/README.md` 开头加一段"目录约定"说明：每个阶段目录是当时运行的完整快照，重复是有意的。
3. **`src/models/sc_inr_adaptive.py`**（388 行，8 个注册别名，docstring 引用不存在的 `configs/registry/models.yaml`）。SC-INR 在论文里只是"实验性变体"，但阶段 17 要加载其 checkpoint，别名可能正是旧 checkpoint 的注册名。建议：保留类与别名，删掉过时的 docstring 段落，补一行说明别名只为服务器旧 checkpoint 服务。不要删别名，除非在服务器上用 `torch.load(...)['model']['name']` 核对过全部 SC-INR checkpoint 的注册名。
4. **`configs/train-div2k/train-liif.yaml`、`train-lte.yaml`、`train-sc-inr*.yaml`**：1000 epoch 原始配方，论文第 2 节的三解码器结果依赖它们。保留，但移到 `configs/train-div2k/legacy-1000ep/` 并在 README 标注。
5. **`studies/two-coordinates/05-matched-encoder/*.yaml`、`07-swinir/*.yaml`** 与 `configs/` 中同名文件只差绝对路径。属于冻结快照，保留；README 注明差异仅为路径。
6. **PLAN.md** 的"下一步"段落仍写"不新增实验"，与 10-29 的方向调整冲突。本轮已改。

### C 档：不要动

- `studies/` 下全部 `FROZEN.sha256`、`*_manifest.json`、`scores/`、`formal/` 等结果与哈希文件。
- 根目录 `datasets`/`models`/`utils.py` 三个符号链接：是让 `import models` 风格的上游代码在仓库根运行的最小代价，改成包结构要改所有冻结脚本的导入，不值得。
- `.gitattributes` 的内部文件排除机制。

### 清理的执行方式

A 档一次提交完成，提交前在服务器上跑一遍 `entrypoints/test.py` 对任意一个现有 checkpoint 和一遍 1 epoch 的 `train.py`（base 配置）确认无回归；这是唯一需要你在服务器上做的验证。B 档按条逐一确认后再做。整体可减少约 900 行代码，不改变任何数字。

## 二、思路评估：现有分析的真实价值

三条发现中，真正有辨识度的是第二条（等步长响应）和第三条（配对来源）。第一条"容量收益集中在 u≳0.75"是本仓库最有用的**设计约束**，但不是惊人的发现：u<0.75 是 LR 本身就带的信息，任何像样的模型都能恢复，这在频域上近乎必然。它的价值在于量化：

- 训练 100→1000 epoch，u∈[1,2) 的 R 提升 0.025–0.046，u<0.75 变化 ≤0.0013。
- 当前 EDSR-LIIF 在 u∈[1,1.25) 的 R 约 0.61–0.65，u∈[1.5,2) 约 0.15–0.37，u≥2 约 0.05–0.2。**头部空间全部在这里。**
- 编码器换 RDN/SwinIR 得 +0.25~0.46 dB；解码器形式（LIIF/LTE/SC-INR/GaussianSR）R 差 ≤0.019；cell 全部信息价值约 0.1 dB。

Codex 的定位（B 类会议分析短文、弱接收）我同意。它对"冲更高层级"的判断也对：需要能改变模型设计或预测未测条件的成果。

## 三、思路评估：当前试点方法（FiLM + u 归一化高频损失）

### 1. 先验上界，用仓库自己的数字算

- **FiLM 臂**。它改变的是解码器读取 cell 的方式。cell 在该配方里的全部贡献是 no-cell 差距 0.09–0.13 dB（表 1）。原生拼接已经提取了其中大部分（否则常数/打乱控制不会恰好等价于无 cell）。FiLM 能多榨出的只能是这 0.1 dB 里拼接没拿到的那一部分，合理预期是 0.02–0.05 dB。**P1 的 0.05 dB 门槛处在这个区间的乐观端。**
- 更根本的张力：**发现 1 本身预言解码器形式不改变 R(u)**。SC-INR 用解析 sinc 做尺度配对响应，设计上比 FiLM 更"物理"，结果落在共享曲线上。FiLM 是又一个解码器形式改动，而我们自己的论文说这类改动不动 R(u)。如果 FiLM 真的显著有效，反而要回头解释为什么它是例外。
- **u 损失臂**。它是训练信号，属于"容量/训练"这一侧的杠杆，与发现 1 的方向一致：把梯度集中到唯一有头部空间的频段。它模型无关，这是它最大的优点。风险是 48 像素窗口的频率分辨率只有 1/48，s≈1 的样本上 u≥0.75 只覆盖 r≥0.375 的少数频点，损失在低倍率样本上近乎稀疏；λ=0.5 也只是按量级估的。
- **wide 臂**是对的控制：若 full 不超过 wide，就说明收益只是参数量。

### 2. 噪声校准

阶段 13 两种子的同臂平衡均值差：EDSR 0.012、no-cell 0.005、tinydec 0.020、RDN 0.027 dB；对比量的种子间散布 0.007–0.015 dB。因此 P1（≥0.05）单种子可信，P2（≥0.03）处于边缘，单种子通过只能算"值得跑 seed 2"，不能写进任何结论。

### 3. 与文献的关系（本轮只做了快速检索，不是穷尽）

- 尺度条件化网络不是新想法：ArbSR 的尺度感知特征自适应、Meta-SR/CUF 的由尺度生成滤波器、CiaoSR 的尺度感知注意力、Thera 的尺度配对热场响应。FiLM-on-cell 作为"方法"新颖度低；其可辩护之处只有"由受控实验（发现 3）推导、参数对齐比较"。
- 频域损失也有先例：Focal Frequency Loss、HiNOTE 的频率感知损失先验、LIWT 的小波域方法。u 归一化（按倍率把频段对齐到 LR Nyquist）是一个小而干净的新点，且与论文的坐标系一致。

### 4. 结论

即便 P1、P2 都过，"EDSR-LIIF 上 +0.05~0.08 dB"也撑不起一篇方法论文；近两年 ASISR 方法稿在相同编码器下报告的增益在 +0.1~0.3 dB，且要对比 LTE/CiaoSR/SRNO/Thera。试点的正确用法是 **go/no-go 和定位哪一半有效**，不是论文主结果。

## 四、如果要"方法突破"，分析真正指向哪里

把三条发现当成约束来推：收益只能来自 u≳0.75；解码器形式不是杠杆；容量和训练是杠杆；cell 的价值在于正确配对。由此有三条路线，按我认为的期望值排序。

**路线 1：尺度配对的训练配方（模型无关的 plug-in）。** 把 u 归一化高频损失做成任何 ASISR 模型都能加的训练项，加上按 u 配对的查询采样（训练时对 u≥0.75 区域加密采样或加权），在 EDSR/RDN/SwinIR × LIIF/LTE 六个组合上验证。卖点不是单点 dB，而是"跨编码器、跨解码器一致地把 R(u) 在 [1,2) 抬高，低频不变"，并用论文已有的 R(u) 工具解释收益落点。这条路和发现 1 完全一致，成本最低（每组 200 epoch 约 17 小时单卡），失败也能解释。开题的"方法"章节可以稳妥地落在这里。

**路线 2：把配对的尺度信息送进编码器。** 发现 1 说瓶颈是编码器容量，而 LIIF 系的编码器完全不知道目标倍率，必须为所有可能的高频段编码特征。把 cell 配对到编码器（对 EDSR 残差块做 FiLM，而不是对解码器 MLP），等价于让编码器把容量集中到当前倍率需要的 u 段。这是 ArbSR 思想在 INR 框架里的重新实现，新颖度中等，但有一个可检验的、来自我们分析的独特预测：收益应表现为 R(u) 在 [1,2) 上升、u<0.75 不变、且随倍率变化而变化（因为同一 r 在不同 s 对应不同 u）。若成立，是比解码器 FiLM 强得多的故事。成本与路线 1 同量级。

**路线 3：显式两路分解。** u<0.75 全部模型都饱和，就把它交给确定性的抗混叠插值，学习路径只预测高带残差，参数全部花在有头部空间的频段。LTE 的 `upinput` 已经部分这么做，而它的 R(u) 仍落在共享曲线上，所以我对这条路线期望最低，除非配合路线 1 的损失一起用。

**不建议再做的：** 继续设计新的解码器形式（新基函数、新的 cell 响应函数）。SC-INR 是这个方向在本仓库里的完整一次尝试，结论已经写进论文。

### 对开题的影响

题目"面向任意尺度图像超分辨率的频谱感知连续表示方法研究"可以保留。三部分结构中，第一部分（分析框架）已完成并有可投稿的稿子；第二部分（方法）建议按路线 1 为主、路线 2 为探索来写，承诺"尺度配对的训练与编码配方"而不是"新解码器"；第三部分（EEG）目前没有数据和基线，开题里写成展望，不写成承诺。

## 五、建议的决策与下一步

1. **本轮先取回试点结果**（你在服务器上把 `runs/claude-20261029-sp-pilot-v1/` 下的 `score.json`、`results.txt`、`score.out`、`probe_*.json` 和 `logs/` 提交到仓库 `studies/method-sp/01-pilot/results/`）。我据此按 P1–P4 判定，并单独看 uloss−base 与 film−base 哪一半在起作用。
2. **无论结果如何**，把"方法"重定义为路线 1（主）+ 路线 2（探索），在 PLAN.md 写明两种解释和预测后再由你冻结协议、跑实验。不在试点之外追加任何训练，直到这一步完成。
3. **清理**：A 档我可以直接做并提交，只需要你在服务器上跑一次回归验证；B 档等你逐条确认。
4. **需要你决定的事**：是否接受"方法 = 训练与编码配方"这一重新定位；A 档清理是否现在做；开题初稿是等试点结果还是现在先写框架。
