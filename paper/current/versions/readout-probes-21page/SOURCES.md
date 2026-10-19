# 论文工作稿：出处与修订状态（2026-10-08）

## 当前论文基线（2026-10-08，导师要求先收敛主线）

唯一现行稿为 [main.pdf](main.pdf)，英文题名改为 **Frequency Recovery and Readout Effects in Arbitrary-Scale Super-Resolution**。前9页是正文和参考文献，10–18页是补充材料；正文结论在第8页。PDF SHA256=`3573e67e5920afa38f2c69c60ff5844c4b672c3dae7e31c5e39b31f47effaba3`。当前稿继承Claude已有结果，不新增网络、目标cohort或GPU训练，也不声称已形成A类新贡献。旧15页完整工程已保存在 [versions/pre-baseline-15page](versions/pre-baseline-15page/)，SHA=`7b14296ac84603fdd949195694824bae2b5e6e09dc9dc4185eb5a75e144bfbf4`；13/14页各阶段归档继续保留。

正文主线统一为：**倍率与频率恢复 → Nyquist归一化 → 成对cell/footprint读出损失 → 尺度平均的预测与失败**。经典Wiener/GP公式是分析工具；G2、稀疏、EEG、开发对照和Gaussian歧义保留完整补充材料，正文保留其对主张的约束。

|处理项|当前口径及直接证据|
|---|---|
|G1元数据过期|独立核查.json顶部原先仍写全文hash pending，现与已存在的frozen_protocol_verified对齐；原字节留档为versions/pre-baseline-15page/g1-独立核查-original.json。root核完整协议SHA 548b26fec4d0d7ed34365915acea3888853f54d5967a811b9a6ebc62a61ca311，所有数值payload不变。|
|L1定量及失败|[保存能量数组独立复算](../补证-20261008/paper-baseline-20261008/evidence_audit.json)核九组u=2sν、pool-then-divide和七箱定义。归一化改善9/9，冻结阈值仅6/9；barbara删除和BSD前30中抽14的200次诊断是事后解释，不能修复原失败或证明JPEG机制。主表全部九值按原数组重算。|
|F1正确SC配对|同一审查核36个保存SC cells与原pooled一致。频带平均L=-.00001421/+.00013676/+.00805506，A均值-.00060908、x3b2+.00103382，保留Set14x2b3负例-.0378833。高相关指排序，不称同量级机制；13图HR256与L1的14图HR240明确区分，差comic.png。|
|E3噪声与参照|[源码审查及记录重算](../补证-20261008/paper-baseline-20261008/e3-audit/conclusion.json)确认σ=.002是prior模拟真实注入噪声；自然图没有注入噪声，ν=1e-5为假设方差/算法正则。generic幅值还用了HR真方差；LR clipping/DC及RGBmean口径明确。27cell及10k逐图ratio bootstrap复算差0，但不是pixel/pooled CI复核，不声称matched-noise训练前边界。|
|贡献与对照|L1/G1/L2支持程度逐级区分，所有失败留正文总表。G2 36/36同选power2.5、相对CV更差更贵；194.6秒静态回放是事后身份核，不含选静态先验成本，不能当确认性廉价选择。稀疏oracle > DIP > generic；EEG760缺失，不作正文跨模态贡献。|
|措辞与图表|撤回safe处方、通用线性上界、架构饱和因果、网络机制、通用采样新定理和编辑历程。新增由L1保存数组生成的六面板尺度恢复图及协议/结论层次表；reader PDF隐藏机器出处标签，完整来源继续在本账本。|

科学审查原件为 [manuscript_review.json](../补证-20261008/paper-baseline-20261008/manuscript_review.json)，所列修正已由root落实。上述复算只用已存记录/能量数组，CPU不足1秒/各120秒预算，无GPU、无新fit。E3缺像素/E/S/真残差、G1缺逐图权重、EEG760缺档、参考阈值敏感性未做，均作为真实证据限制保留，并非补证完成。若日后要恢复对应强主张，须另定有限回顾性敏感性或新协议，不将本次措辞修正当新实验。

TeX/BibTeX在research-175新隔离根编译成功；18页全部100dpi渲染审阅，重点页单独检查；未解析引用/LaTeX warning/Overfull均0，Underfull 8项保留。当前哈希与逐文件SHA见 [编译核查.json](编译核查.json)；源核与本轮验收见 [baseline-verification.json](../补证-20261008/paper-baseline-20261008/baseline-verification.json)。这是可审阅研究稿，不是导师验收、投稿或录用。当前持续安排保持PAUSED；先审此基线，再单独讨论创新贡献。

## 以下为早期修订与历史证据记录


编译：`pdflatex main && bibtex main && pdflatex main && pdflatex main`。正文 `\src{键}` 对应下表；现行阅读稿已设 `\showsrcfalse`；需要审计标签时可临时打开，来源以此账本为准。

| 键 | 出处文件（/mnt/project-files/ 下） |
|---|---|
| OUT3/d8 | knowledge/claude/主旨与大纲-v1-分析型论文-20260929.md §3（事前三区地图） |
| OUT7.1/h6c | 同上 §7.1（单图 ×4 网络 vs 线性上界） |
| OUT7.2/sp_scan | 同上 §7.2（稀疏采样率扫描） |
| OUT7.3 | 同上 §7.3（EEG 标定集） |
| E2 / E2§n | knowledge/claude/结果-E2-谱失配代价-20261003.md |
| E3 / E3-key | 结果-E3-单图多倍率-20261005.md |
| E1 / E1§n | 结果-E1-多帧全扫-20261005.md |
| F1, F1-corr2 | 结果-F1-读出算子价值与三区-20261005-20261004T223548-0264.md（更正 2：SC 侧正确配对） |
| E4-T1 / E4-T2 / E4-E3sens | 结果-E4-总表-20261005.md 表 1 / 表 2 / E3 敏感性 |
| DER§n | paper/derivation/推导修正-v1-20261005.md 第 n 节（推导线程） |
| D11 | 结果-D11-E1切换提前诊断-20261005.md |
| L1L2 | 结果-L1L2-尺度坍缩与读出算子-20261005.md（预注册同目录） |
| G1 prereg | paper/derivation/预注册-G1-读出损失与足迹增益偏差-20261005.md |
| LIT-B | knowledge/claude/文献调研与路线定基调-20260923.md 第一部分 B |

## 2026-10-11：收紧主线与前两阶段完成

正文仅保留ASISR频率恢复和成对读出两项贡献，明确A/Q/c及预训练模型与逐图拟合不同；经典分析和长guidance进入补充。旧18页工程存versions/pre-readout-20261011，收紧后19页未评分稿存versions/tightened-pre-probe-19page。当前21页强化稿（正文及参考文献1–8，补充9–21）加入固定权重诊断附录C和两张完整结果表，不升级为新机制贡献。当前PDF SHA81f3cf179c2a18c6ebfe9f75a1862a3a59b31a0909e60aa121ddd39ebce5505b，完整QA见编译核查.json。

直接近邻补[IPE-LIIF](https://arxiv.org/abs/2112.05756)，核积分编码与额外cell对照；[LIIF](https://arxiv.org/html/2012.09161)已有固定查询cell干预；[Thera](https://arxiv.org/html/2311.17643v4)已有解析Gaussian读出；[HyRA/FSDS](https://arxiv.org/html/2405.07919v1)有响应分解。积分/cell干预不作首创。G1免费band-index控制四组同秩同Spearman，不重复旧推理。

|键|原件与边界|
|---|---|
|READOUT-20261011|../补证-20261011/readout-probe-v4/protocol.json、experiment.py及审查；完整原件../补证-20261011/collected/codex-20261011-readout-formal-v4/，冻结模型与四份checkpoint在同层codex-20261011-readout-probe-v4/snapshot/。协议d0897df4、源码38d3bee3，72合成输入+两倍率barbara，148输入、152预测全cohort SHA8e625539051d125de06d1790a3bc116f8f82366e251030b74be4971aa49483b3先冻结才评分；576 synthetic+24 natural rows完整。|
|READOUT-AUDIT-20261011|readout-audit-v1.py留原版本，readout-audit.py（v2 SHAa7d8f2ce）和collected/codex-20261011-readout-audit-v2.json。全部输入/预测、41源记录、四checkpoint、像素评分/四组指标/gates/完整成本重算通过，最大差1.78e-15。18固定k/direction簇2000bootstrap仅网格敏感性，非自然泛化CI；自然输入生成未独立重做，输出像素重评不代替网络重实现。|
|READOUT-CROSS-20261011|readout-cross-analysis-v1/protocol.json、analysis.py及pre-score-freeze.json（08:38:00Z先于08:47:18Z预测冻结与首次评分）。源码97a60367、协议ea182310；collected/codex-20261011-readout-cross-run-v1/含完整2304rows与576target scores。独立复核62dadf5d核2304rows/8groups/96pairs通过，最大差3.55e-15。无新增推理/拟合，只是目标真值参与的描述性评分。|
|READOUT-INTERPRETATION|../补证-20261011/paper/probe-interpretation.json核完整下载SHA/300输入输出/源与checkpoint；额外same-Q hint sign反向次数为评分后的描述性计数，不增加门槛。旧F1 barbara SC六band比较最大差5.76e-6，CPU/GPU数值路径不同，不称bit复现。|

正式CPU4 nice10 GPU0，无训练，预先2400秒绝对/累计预算内1376.925秒完成（完整launcher1377.305秒），CPUuser/system4501.156/601.088秒，RSS748848KiB，48,709,632 RGB查询。预测冻结08:47:18Z；源码/协议/所有原件核与归档一致，无失败正式case。独立审查1.287秒，cross评分.704秒，独立cross复核.626秒，各自120秒有界新根；完整成本日志保留。

SC积分forecast相对band MAE点估计改善24.249%/22.419%，相对affine footprint46.032%/42.654%；但负收益召回0/9与1/11，vsband网格敏感性区间均跨0。LTE两组不满足共同门槛。两个oracle候选均0/4完整group通过。全部568/576数值resolved，8未分辨行不删。cross held-phase matched最佳18/21/19/16（各72），匹配hint各band均正负；这不与未定义random policy比较，也不单凭匹配差证明非积分。

barbara SC×2 b3重训差-.193296，而同权重native-vs-zero-cell +.009733；×2 b2同权重-.006248=alignment-.002765−change energy.003483。这是原件核算和事后分解，不是负收益前瞻机制。合成内容/rectangular/zero-cell训练支持限制，实际LTE无原文cell clamp，积分对象只是zero-cell条件参考场；自然HR是离散真值。结果不足以启动匹配训练或宣称A类贡献，留待导师决定。安排保持PAUSED。

完整下载归档SHA=c3a21fd1b3f24d7f398113f24c271f22fca6173650866122877bb597aa54642c，位于../补证-20261011/collected/codex-20261011-readout-complete-v1.tar.gz。v1底层崩溃、v2独立20.51秒预检、v3缺time包装退出与v4版本均保留；未覆盖或重启旧checkout/训练。当前没有运行中的本任务CPU/GPU推理。

## 原稿占位清单（2026-10-05历史，当前状态见下）
1. §5.4 E1：D11 结果已填（a 点估计成立、b 0.856 成立、c 不成立、e 无区分力不作证据）；已按推导修正 v1（D11 后版）统一措辞：判据 C(Q) 只能排序，MTF 零点机理撤回。
2. §5.5 F1：SC-INR 一侧已用正确配对 seed1 数字（H-F1 0.400、H-F2 0.492、跨架构 0.985），A 区 L 具体值笔记未给；ε_b² 只作频带顺序预测量（增益失配机理已被 L2b 否定，不写），G1（seed2/3）结果待填。
3. §5.7 E4 预测 vs 实测总表：等 1、2 完成后抄。
4. 局限中三区阈值 0.1/0.8 的敏感性分析：未做。
5. 结论节、图 1（三区热图，需 d8 JSON 出图）。

## 原稿未核项（历史记录，不作为当前审批要求）
- SC-INR、Rot-E 的引用条目与一句话描述（§1、§2、§5.4 的臂描述）。
- refs.bib 中标 TODO 的条目（WINNER、SEC、WIEN-INR、SuperF 等）作者与出处未核实。
- 推论 1（R=1/2 翻转）已由推导线程核对（逐频率、对称倍数失配下精确成立），正文已写明适用条件与稀疏 10% b1 反例。
- EEG 只用了标定集 200 段的数字，H8 测试集 760 段待补。

## 写作中刻意避开的说法
- 「读出算子价值由事前 R_lin 预测」（F1 主判据不成立）。
- 「nosinc 只在 b3 偏低约 0.02」：h6c 中 SC-INR-eq 与 nosinc 也换了编码器，属同一配对问题，未使用。
- β² 定律数字：锐利子带复核后常数已改（3.67%），正文未引用。

## 2026-10-08 本地修订（原件保留）

当前工作稿继承完整 Claude LaTeX 工程；原件在 ../Claude原稿-20261007，不改写原始数据。源码、原始下载哈希和科学边界分别核查。

|新键|直接证据|
|---|---|
|E2r|../服务器结果-20261007/claude-20261005-e2r/e2r_result.json；独立离线核查见同层核查.json|
|CPU-10000|../补证-20261007/cpu-checks/d11_bootstrap_10000.json 与 scalar_formula_and_band_aggregation.json；root 再执行确认同值|
|G1-20261008|../补证-20261008/g1：8次原脚本 GPU 推理、原 ana_g1 输出与独立核查；冻结原文 SHA256=548b26fec4d0d7ed34365915acea3888853f54d5967a811b9a6ebc62a61ca311|
|L2-20261008|../补证-20261008/multiseed：新增 seed2 SC / seed3 LTE 原脚本 GPU 推理，与两组历史数组合并，独立按4组≥3门槛判断|
|G2-protocol|../补证-20261008/guidance/protocol.json、预注册-分析指导选择-v1.md、源码SHA256.json；运行根各自生成frozen.json；v1/v2/v3开发数值失败原位保留，v4双精度PCG最多4000步、真实残差1e-5及预先24h预算；正式完整结果见G2-result|
|G2-result|../补证-20261008/guidance-formal-v4/summary_target.json、evaluation/、selections/、selection_cohort_target.json、resources.json、COMPLETE；36/36完整结果，全部选择先冻结再评分|
|CTRL-20261008|../补证-20261008/phase-linear-controls-v1/protocol.json、analysis-plan.json、源码SHA256.json、审查-agent.json；../补证-20261008/phase-linear-controls-run-v1/scores.json、prediction_cohort.json、resources.json、独立对照复核-agent.py/.json、收集来源.json。24固定开发解全预测先SHA冻结后评分；tar165f93a2/cohort61dc04e4/sourcea70ff66b/protocol19c8942b；17.39秒CPU、0失败。另旧inr-phase-development-run-v1保存FF12fit。经典方法/两已见图/单seed/固定mean与ridge，不作为新算法或机制。|
|CAL-20261008|../补证-20261008/common-law-calibration-v1/protocol.json、analysis-plan.json、审查-agent.json、源码SHA256.json；../补证-20261008/common-law-calibration-run-v1/完整384校准风险、8静态推荐、24开发评分、132pair、审查.py/.json、收集来源.json。sourcee1615680/protocolba16fe05/cohortb11ea98f/tar0b90f255；600秒预算内8.2328秒CPU完成，独立5.6723秒全量复核、最大差4.44e-16。15/16推荐吻合但15/132排序反例，6个phase×图推荐均等于固定RBF sigma3；已见开发图回顾性、有限32图joint RGB经验law，不称逐图适配/总体保证/新理论。|
|SEP-20261008|../补证-20261008/task-separation-development-v1/protocol.json、theory-agent.json、mixture-agent.json、prior-art-agent.json、numerical.py、结论.json；../补证-20261008/task-separation-check-run-v1/frozen.json、rows.json、resources.json、COMPLETE。原协议b81e8cdd不改，源码44f2591c；36确定性矩阵检查0失败、gap残差≤1.37e-15，独立手推核所有常数/类限定。phase02 unrestricted regret精确2theta²/a，restricted二选一4theta²/a；混合估计对照必须保留。仅正定8点known magnitude/unknown sign/same-geometry sampled calibration+fresh eval、trace risk，非HR校准/自然图/INR。原件和负新颖性结论保留；附录E仅经典示例，不能称新定理。|
|G2-audit|同根独立图像复核.py/.json、独立统计复核-agent.json、收集来源.json；完整归档SHA验证、78缓存输出从RGB重评、12图簇10k bootstrap/成本独立重算；非缓存候选只核保存MSE/证书，未另跑GP|

修订撤回 generic-prior 对任意图像的普遍线性上界、频带 Jensen 单向误差、网络 Rnet 端点必然无消融损失及架构饱和的因果解释。新增矩阵风险正交性证明、等能量双模反例、G1/L2 完整四组表、D11 协议级 bootstrap 和结论。E2 原版表格与独立重写数据分开。

图1由 figures/make_figures.py 直接计算经典标量公式与反例，不取目标数据。

原稿占位处理：已补SC正确配对A区均值、G1及L2四组结果、结论和解析图；E4旧汇总不再转抄，确认性判断用相应原结果。SC-INR为当前研究实现，Rot-E为本实验臂，未伪造公开论文引用。三区0.1/0.8阈值仍是操作划分，未称稳定自然边界；EEG测试集760段仍缺独立原件，不当确认性证据。G2完整结果与反例已写入§5.6、摘要、贡献、局限及结论；开发数值失败继续保留。

G2执行更新：v4 full1+mini6开发通过后，2026-10-08 02:58正式36case启动，07:01结束，输出新根guidance-formal-v4。全cohort SHA=64d2053e前缀；源码与预先冻结一致，36/36完成、无失败。归档SHA=2a66b177前缀，完整原件本地保存。2592份求解证书全部通过，最大3134步；主方法78份缓存输出独立像素复评最大MSE差8.7e-19。

后续研究定位（合成歧义和二次型候选仍未作为正文贡献；同观测开发对照已入14页附录D）：导师授权沿本主线增强A类主会竞争力，持续安排曾恢复ACTIVE；当前关口已关闭，状态PAUSED。原13页PDF及源码完整存versions/g2-13page，SHA=b422b624b7d0ab816745efd214c586f2eea234784a7bd5078533c9480fd4280d。当前14页PDF新增开发附录/局限，摘要主贡献不升级，hash与全页QA见编译核查.json。新增证据均处于研究/开发阶段，不作已获创新。

- [合成冻结协议](../补证-20261008/ranking-ambiguity-v1/protocol.json)、版本源码及theory.json；v2 SHA=21163b04/0c7cd9c9，preformal-v1无正式运行并原件保留。
- [1920case原件与独立核查](../补证-20261008/ranking-ambiguity-formal-v2/)：完整选择SHA=9188cc29前缀后评分，tar SHA=e2c8df01前缀；1887有效、33空集、0数值失败，197认证未见错、真谱覆盖1841。全部LP端点独立重求，成本/拒绝/反例保留。仅已知均值、trace8的Gaussian谱族fresh-draw总体风险，非单图/同y安全；EB强、认证低，不称实用优势。
- [二次型分析](../补证-20261008/ranking-ambiguity-v1/quadratic-gap-theory.json) SHA=c7915121前缀：固定L风险差、方差、保守界及非可估偏差区间；[经典覆盖核查](../补证-20261008/ranking-ambiguity-v1/prior-art.json)确认MINQUE、协方差分量可估性、compressed covariance、partial identification和ENSURE已有工作。不是新估计原理，界可能很宽。
- [实际FF开发协议/源码/启动](../补证-20261008/inr-phase-development-v1/)：原Claude e1ff SHA=0a82c9fa，AST仅FF/gauss_B。源码32fc0534、协议fca50a58；两已见图0801/0802×三同成本水平相位×sigma3/10、2000步、1seed=12fit。单A800 cap15%、nice10、4线程，2h绝对/累计预算；13:01:04启动、13:01:59结束，无旧训练重启。
- [12fit完整原件](../补证-20261008/inr-phase-development-run-v1/)，tar SHA=bad71f5b前缀，权重/预测/观测/损失/成本保留；[独立开发复核](../补证-20261008/inr-phase-development-run-v1/独立开发复核-agent.json)核12×2000步、源/数据/坐标/权重/预测/资源通过，PSNR差<7e-7dB；每几何重置seed1001形成共同随机数耦合。开发[0,2]全部优于[0,1]、sigma3全部胜sigma10，不能将合成谱可辨识秩解释成实际网络PSNR机制。水平point模型与旧integrated E1前向不同，仅为开发；尚未建立实用选择、二维谱恢复或正式未见图像结果。

24组同观测对照现已入附录D与表12，保留固定RF的3组反向和sigma10严重raw bias。各模型自身方差不是统一真实risk，最近点距离相同也不能解释差别。共同校准law开发诊断已完成并全量复核，见CAL-20261008。其推荐仍退化为强静态RBF sigma3，保留全部排序反例，不能称新方法；下一步先过有界理论/新颖性关口，再决定是否有理由冻结未见cohort。未证明新颖性或A类主会竞争力，不把负结果删去后堆实验。

完整结果：分析法相对默认+1.105081dB、图像簇95%CI[.726306,1.504180]、严重退化0/36，预定默认改善门槛通过；CV+1.216655dB。分析−CV=-.111574、CI[-.312918,-.002450]，8466.8秒对4821.7秒；优于CV且不增加成本的门槛失败。分析36/36均选power2.5；固定power2.5会产生相同输出且公共准备+最终重建194.6秒，仅为事后身份核查，不升级为新增确认性baseline。结果不能证明逐图适配增益或廉价选择。冻结汇总器在协议CV门槛之外还要求无选择失败；本次无失败，判定不受影响。

图2由figures/make_guidance_figure.py从完整evaluation与独立核查绘制，全部12图呈现，源hash见guidance-results.sources.json。本文成本为CUDA同步后的阶段墙钟，非GPU忙时或独占设备benchmark；完整资源账本与政策成本分开，不重复计共享输出。当前PDF的编译/源码哈希及全页QA见编译核查.json。

近期引用按官方原文核对并补齐原稿TODO：

- [FreSh](https://arxiv.org/html/2410.05050)：匹配未训练输出与目标谱，也评测NeRF逆问题。
- [WINNER](https://arxiv.org/html/2509.12980)：目标谱质心设定初始化权重的高斯扰动尺度，包含去噪任务。
- [SEC](https://arxiv.org/html/2605.12709)：校准集近邻返回网格搜索配置，不保证新目标最优。
- [WIEN-INR v2](https://arxiv.org/html/2509.15494v2)：2026-06-08修订题为Multi-resolution Enhancement for Full Spectrum Neural Representations，采用小波子网；不是仅傅里叶频带配置。
- [GSASR](https://arxiv.org/abs/2501.06838)：完整四位作者，ICCV2025。
- [Thera](https://arxiv.org/abs/2311.17643)：TMLR2025，纠正原稿CVPR出处。
- [低通与FSDS](https://arxiv.org/abs/2405.07919)：完整六位作者，ICML2024。
- [SuperF](https://arxiv.org/abs/2512.09115)：补齐作者，2025预印本。
- [Nasir基准 v3](https://arxiv.org/abs/2601.17723v3)：2026-08-25标题改为Empirical Benchmark。
- [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/)及BCI2000：数据集版本与原系统引用分别陈述。

不再把FreSh/WINNER概括为仅充分观测拟合，也不以经典LMMSE或全部先行工作没有逆问题作为新颖性。本文新增贡献须落在经过证据支持的任务比较、失配诊断和指导可靠性范围。

本轮共同校准结果已写入附录D，当前仍14页，PDF SHA=ad1cb9b29e451c1d55b329198e5dd9e4c3f2474392cf4654dde3138668741561，服务器TeX成功、14页全渲染与更新页150dpi核查通过，LaTeX warning/overfull/未解析引用均0，Underfull8保留。此前14页对照稿及完整源码在versions/controls-14page留档（SHA6cb25640），13页G2稿继续保留。

新颖性边界补充：[Zimmerman2006](https://onlinelibrary.wiley.com/doi/abs/10.1002/env.769)已直接比较已知协方差预测、协方差参数估计及其混合设计；泛称“重建与学谱采样目标不同”不能作为新贡献。[Mishali–Eldar](https://arxiv.org/abs/0709.1563)的未知带位置采样代价属稀疏完美恢复模型，不能直接套入Gaussian风险。下一次仅按[有界理论协议](../补证-20261008/task-separation-development-v1/protocol.json)核具体8点风险分离与一般alias-block sharp界，候选公式尚未证明；一次1800秒root主动墙钟、2026-10-08 07:00Z绝对截止，不延长，不启动GPU。如果仍只得到经典两点判别+例子，应结束该候选新理论包装并保留负结果。

2026-10-08 14:49本轮理论关口闭合：固定8点正定谱族，同四测量与4(K+1)总预算，oracle重建偏好02，但未知sign时足够K的01可超过02所有估计器的最坏total risk；严格区分restricted选择4theta²/a、unrestricted混合2theta²/a，不能把前者下界外推给所有估计器。原protocol未改，完整手推与36矩阵检查见SEP-20261008；不是一般alias-block sharp定理，经典robust Bayes/两点检验/采样设计已有覆盖。特定公式未检索到完全同构文献不等于创新。新引用Kitagawa2011（working paper，AppendixB原文）与Zimmerman2006（全文），Kipnis等arxiv1405.5329的known-spectrum alias-class MMSE在prior-art原件限定陈述。

当前15页强化工作稿新增附录E及action-class局限，PDF SHA=7b14296ac84603fdd949195694824bae2b5e6e09dc9dc4185eb5a75e144bfbf4；服务器pdflatex/BibTeX成功，15页全渲染与14/15页150dpi审阅通过，warning/overfull/未解析引用0，Underfull7记录。前14页校准稿完整存versions/calibration-14page（SHAad1cb9b2），更早controls-14page与g2-13page均保留。当前候选未通过A类新贡献门槛、未提交，按预定停止条件结束自动重复推进；持续安排置PAUSED，目标未称达成。若继续A类目标，需要先重新明确具有非经典增量的核心问题，不自动扩图像/重训或另换新网络路线。

2026-10-08 18:28按导师请求，已在Claude现有「工作如何进一步打磨」线程上传上述15页PDF，并补发完整G1/L2/D11/G2、合成认证、FF开发、线性控制、共同校准law和八点理论关口进展，包含负结果、成本、证据路径及适用范围；界面确认完整消息已发送，Claude已开始只读核原件及独立复算。请求审查主贡献/新颖性、理论与实验漏洞，以及一个有明确增量和停止判据的A类候选问题；限定本轮评审，不启动GPU或修改冻结成果。Claude首轮仅基于PDF的初评建议将主线收回ASISR、压缩审计细节、移出开发附录，并提出阈值依据、barbara诊断、SC配对量级、EEG缺件和E3噪声口径等待核问题。这些属于外部评审建议，尚未由Codex重新核实或采用；完整进展送达后的评审仍在进行。线程标识session_018KquTKJ9K45VcYptoikUeu。

2026-10-08 18:44导师要求继续跟进同步。已读取Claude18:31完整初评和18:38原件调取状态：PDF及共享旧笔记已读、八点常数独立复算；当时新增实验仅确认收到修订目录/SOURCES/补证目录树/G2汇总，其余待核，不能称完整原件复核已完成。已在同名现有线程补发七类证据的精确小JSON/协议/成本/独立审查/逐case路径，补齐服务器根路径，并要求逐项报告已读/待到/缺失和核查层级。另指出194.6秒只是事后固定输出身份成本，未计预先静态校准策略成本，不能据此确认性称快43倍；经验强模型差距不能证明Bayes极限或识别容量因果；.5与κ→∞时约.382不能自动替代操作分区阈值或改写旧冻结判据。Claude初评认可八点候选收窄、认为当前无A类突破，建议主线聚焦ASISR经验诊断；其强模型建议仍待先行工作/共同任务/风险下界核查，未采用、未新启实验。补充消息界面完整送达，等待新增原件核查反馈；当前界面线程键cmsg_01VW46txaMf4tdNbMTu7w2tQFwvVZD5ZNnTiQMMYPtbPXz。

2026-10-08 18:46–18:49同步闭合更新：Claude回复先前未到的46份文件及evaluation/36项现已只读核过，无缺件；核查主要是summary/protocol/独立核查JSON字段与hash，未重新逐case计算所有实验，三份ranking理论文件仅核hash，跨对象引用仅关键词检查而非正文全文审查。以上是Claude自报阅读层级，不升级为另一套完整独立复现。Claude接受事后静态成本不构成43倍确认性优势、阈值不能改写冻结判据，并撤回“贝叶斯极限/强模型推理”作为A类候选；贡献判断未变，建议保留合成与反向结果作为边界证据。

本轮核实的待修正项：g1/独立核查.json顶部status/protocol_original保留早期“原文待核”而尾部frozen_protocol_verified已记精确hash，属于状态接续不一致。Codex本轮重新读取尾部path指向的完整G1原预注册文本，SHA256现场确为548b26fec4d0d7ed34365915acea3888853f54d5967a811b9a6ebc62a61ca311，并于18:49同步Claude；不把旧头当作当前原文缺失。论文intro.tex的safe需在后续修订中收窄；本轮未改正文/原实验或重新编译。为应对当时缺件准备的[64份原件归档](../补证-20261008/claude-review-sync-20261008/claude-research-review-originals-20261008.zip)（SHA=f56242a1de39aac2b6ed9cbc56d802681d0d0ed55b1a2b0aeb8353b8cf7a7d7e）及manifest/receipt保留，zip逐文件字节hash验证通过；Claude随后确认原件到齐，因此未重复上传。该包仅传输准备留档，不是新结果或当前主本副本。
