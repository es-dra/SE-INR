# SC-INR 路线正式收口（P1.12-C）

日期：2026-08-07  
状态：`closed-as-new-method / keep-as-internal-case-study`

## Node Contract

- 本节点回答的问题：哪些 SC-INR 材料还能作为内部可复现事实，哪些活跃入口必须停止暗示可投稿的
  新方法 claim。
- 主检查项：README、论文入口、模型 taxonomy、artifact 白名单、claim matrix 与 current state 是否
  对路线状态给出一致答案。
- 成功标准：第一次进入仓库的研究者会先看到“路线已停止”；训练命令只标为冻结复现；论文草稿明确
  不是投稿稿；性能数字、代码、checkpoint 和 raw result 保持不变。
- 失败标准：任一活跃入口仍把 SC-INR 称为最终主方法，或暗示补一个相邻 baseline/factorial 即可自动
  恢复 novelty；收口过程中覆盖用户内容或删除不可恢复资产。
- 禁止事后调整：不修改 benchmark 数字，不删除模型代码、checkpoint、raw result、图片或用户论文
  内容；不启动 IPE/Thera、factorial、metric、descriptor、模型分支或 GPU 实验。
- 预期产物：本决策、`paper/README_zh.md`、最小活跃入口修订、daily memory 和 context capsule。
- 独立审查：真实 skeptical paper reviewer、repository-entry reviewer、method/formula reviewer。

## 普通语言解释

SC-INR 实际做的是：从 LR feature 预测局部 coefficient、frequency 和 feature-conditioned phase，cell
不进入 learned phase/frequency，而是在进入 MLP 前乘一个解析 box-sinc response。这个 response 会随
输出 footprint 改变 decoder 输入，但后续还有非线性 MLP、local ensemble 和 bilinear residual，因而
最终 RGB 不是解析的 box integral，也没有 whole-network scale equivariance 保证。

现有三 seed 结果只说明：在当前 bicubic 协议下，SC-INR 相对 LIIF/LTE 的 OOD/ALL PSNR mean delta
为正，约 `0.03--0.05 dB`，且与 seed 波动同量级。NoSinc、cross-grid agreement、cell intervention 和
selected crop 都是机制或案例诊断，不能证明 sinc 是增益原因、真实 observation model 更正确或方法
具有稳定尺度外推优势。

最强平凡解释是：pre-MLP sinc 只是一个随尺度变化的幅度门控/正则，与 signed frequency、feature
phase、容量和优化差异共同产生小幅增益；selected texture crop 放大了局部观感。现有证据无法排除
这个解释。

## 为什么现在收口

P1.12 预先要求 IPE-LIIF 和 Thera 两条 direct baseline 同时满足来源、实现语义和公平协议门槛。
结果是：IPE-LIIF 的论文公式可实现，但没有可核验的作者代码、commit、许可证或 checkpoint；Thera
有官方 Apache-2.0 代码，但 checkpoint provenance 不可哈希，原生 5M-update/TTA/TV/AdamW recipe 又
不是 LTE-style 同预算 factorial。双门失败，因此 P1.13 没有启动。

这阻断的是 SC-INR 新方法路线，不是一个性能排名结论。没有跑公平 direct baseline，所以不能反向
声称 IPE-LIIF/Thera 更强，也不能说 SC-INR 完全没有窄实现价值。

## 保留、退役与恢复条件

### 保留为 canonical

- 模型代码、配置、seed1--3 checkpoint、raw result 和 canonical benchmark。
- 已登记的机制诊断、用户确认图片、模型名/raw-key 映射和 provenance。
- `paper/draft/`，但身份改为冻结内部技术记录。

### 退役用途

- `SC-INR` 作为“最终主方法”或活跃论文候选的身份。
- strict scale equivariance、exact box-integrated RGB、通用 sampling-consistency principle、稳定尺度
  外推优势、sinc 因果和投稿 novelty。
- 通过继续补相邻 baseline、metric、descriptor 或 selected figure 自动恢复 claim 的路径。

### 可能重开，但不会自动恢复 claim

只有 IPE-LIIF 作者 source/确认，或公开独立实现通过公式、`radius=rel_cell/2`、原论文趋势与 LIIF
parity 核验，同时 Thera checkpoint 获得 immutable revision、SHA256 和 variant 映射后，才允许新建
Node Contract 重开 P1.12。来源门通过后仍需同协议 direct baseline、model-native/equal-update 分开
报告、multi-seed LTE factorial、有效 observation 正负控和 paired uncertainty，才能重新讨论窄 claim。

## 独立审查综合

- 采纳 skeptical reviewer：直接修改 `main.tex` 的 future-tense 恢复叙事和过强结果措辞，不能只在
  ledger 标冻结。
- 采纳 repository-entry reviewer：根 README 首屏、`paper/README_zh.md`、taxonomy 和白名单必须先
  显示冻结状态；训练命令保留但降为默认不运行的历史复现入口。
- 采纳 method/formula reviewer：身份收缩为 LTE-specific decoder inductive-bias case study；把
  `scale-decoupled observation` 降为“cell 不进入 learned phase/frequency，但保留显式 response”。
- 不采纳删除 `paper/draft/main.pdf`：当前工作树尚未形成可恢复提交，删除不会增强科学边界，反而
  增加误伤用户产物的风险。PDF 以显式冻结标识保留。

三类审查的共识是：现有实现事实可复现，方法身份和投稿叙事必须停止。没有关键分歧改变判决。

## Artifact Triage

- `promote`：本收口决策、`paper/README_zh.md`。
- `keep-canonical`：代码、配置、checkpoint、raw result、三 seed benchmark、登记诊断、用户确认图、
  claim/whitelist/taxonomy 与历史草稿。
- `mark-frozen`：`paper/draft/`、SC-INR 训练配置和训练命令。
- `archive-with-index`：本节点无新增材料需要归档。
- `delete`：本节点没有能证明安全且有价值的删除项。

## 科学进展分类

本节点属于 `strategic negative closeout / evidence-boundary correction`。它没有产生新模型、性能提升或
尺度等变理论突破；价值在于停止一条已连续两次未通过正式门槛的路线，并让代码、结果、文档和 claim
重新自洽。
