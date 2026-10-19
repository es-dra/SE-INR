# 论文工作稿：出处与修订状态（2026-10-08）

编译：`pdflatex main && bibtex main && pdflatex main && pdflatex main`。正文 `\src{键}` 对应下表；终稿在 main.tex 改 `\showsrcfalse` 隐藏。

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
|G2-audit|同根独立图像复核.py/.json、独立统计复核-agent.json、收集来源.json；完整归档SHA验证、78缓存输出从RGB重评、12图簇10k bootstrap/成本独立重算；非缓存候选只核保存MSE/证书，未另跑GP|

修订撤回 generic-prior 对任意图像的普遍线性上界、频带 Jensen 单向误差、网络 Rnet 端点必然无消融损失及架构饱和的因果解释。新增矩阵风险正交性证明、等能量双模反例、G1/L2 完整四组表、D11 协议级 bootstrap 和结论。E2 原版表格与独立重写数据分开。

图1由 figures/make_figures.py 直接计算经典标量公式与反例，不取目标数据。

原稿占位处理：已补SC正确配对A区均值、G1及L2四组结果、结论和解析图；E4旧汇总不再转抄，确认性判断用相应原结果。SC-INR为当前研究实现，Rot-E为本实验臂，未伪造公开论文引用。三区0.1/0.8阈值仍是操作划分，未称稳定自然边界；EEG测试集760段仍缺独立原件，不当确认性证据。G2完整结果与反例已写入§5.6、摘要、贡献、局限及结论；开发数值失败继续保留。

G2执行更新：v4 full1+mini6开发通过后，2026-10-08 02:58正式36case启动，07:01结束，输出新根guidance-formal-v4。全cohort SHA=64d2053e前缀；源码与预先冻结一致，36/36完成、无失败。归档SHA=2a66b177前缀，完整原件本地保存。2592份求解证书全部通过，最大3134步；主方法78份缓存输出独立像素复评最大MSE差8.7e-19。

后续研究定位（合成歧义和二次型候选仍未作为正文贡献；同观测开发对照已入14页附录D）：导师授权沿本主线增强A类主会竞争力，持续安排已恢复ACTIVE。原13页PDF及源码完整存versions/g2-13page，SHA=b422b624b7d0ab816745efd214c586f2eea234784a7bd5078533c9480fd4280d。当前14页PDF新增开发附录/局限，摘要主贡献不升级，hash与全页QA见编译核查.json。新增证据均处于研究/开发阶段，不作已获创新。

- [合成冻结协议](../补证-20261008/ranking-ambiguity-v1/protocol.json)、版本源码及theory.json；v2 SHA=21163b04/0c7cd9c9，preformal-v1无正式运行并原件保留。
- [1920case原件与独立核查](../补证-20261008/ranking-ambiguity-formal-v2/)：完整选择SHA=9188cc29前缀后评分，tar SHA=e2c8df01前缀；1887有效、33空集、0数值失败，197认证未见错、真谱覆盖1841。全部LP端点独立重求，成本/拒绝/反例保留。仅已知均值、trace8的Gaussian谱族fresh-draw总体风险，非单图/同y安全；EB强、认证低，不称实用优势。
- [二次型分析](../补证-20261008/ranking-ambiguity-v1/quadratic-gap-theory.json) SHA=c7915121前缀：固定L风险差、方差、保守界及非可估偏差区间；[经典覆盖核查](../补证-20261008/ranking-ambiguity-v1/prior-art.json)确认MINQUE、协方差分量可估性、compressed covariance、partial identification和ENSURE已有工作。不是新估计原理，界可能很宽。
- [实际FF开发协议/源码/启动](../补证-20261008/inr-phase-development-v1/)：原Claude e1ff SHA=0a82c9fa，AST仅FF/gauss_B。源码32fc0534、协议fca50a58；两已见图0801/0802×三同成本水平相位×sigma3/10、2000步、1seed=12fit。单A800 cap15%、nice10、4线程，2h绝对/累计预算；13:01:04启动、13:01:59结束，无旧训练重启。
- [12fit完整原件](../补证-20261008/inr-phase-development-run-v1/)，tar SHA=bad71f5b前缀，权重/预测/观测/损失/成本保留；[独立开发复核](../补证-20261008/inr-phase-development-run-v1/独立开发复核-agent.json)核12×2000步、源/数据/坐标/权重/预测/资源通过，PSNR差<7e-7dB；每几何重置seed1001形成共同随机数耦合。开发[0,2]全部优于[0,1]、sigma3全部胜sigma10，不能将合成谱可辨识秩解释成实际网络PSNR机制。水平point模型与旧integrated E1前向不同，仅为开发；尚未建立实用选择、二维谱恢复或正式未见图像结果。

24组同观测对照现已入附录D与表12，保留固定RF的3组反向和sigma10严重raw bias。各模型自身方差不是统一真实risk，最近点距离相同也不能解释差别。下一步先在统一独立校准law下检验线性失配风险，经典校准工具不称新方法；形成可检验预测后再冻新cohort及强基线。未证明新颖性或A类主会竞争力，不把负结果删去后堆实验。

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
