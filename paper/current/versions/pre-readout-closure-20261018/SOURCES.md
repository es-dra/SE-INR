# 论文工作稿：出处与修订状态（2026-10-18）

## 2026-10-18：本版独立审稿复核与最小剩余项

本次按导师要求只评审当前PDF `c94877fb`，未改主稿、未运行新模型或训练。根agent重读全文/主图及原始评分，并作独立审稿子任务；主结论与数据基本对应，主要风险是贡献显著性及“why”问题大于HR辅助失败分类实际回答。平方误差恒等式、平均/个体差异、参照改变导致标签改变不能单列为新原理。IPE原文§4及§5.3已明确cell不保证改善及大倍率负效果（https://arxiv.org/html/2112.05756v1）；本文增量应落实到固定权重、同倍率/频带下的损害分布、重复性及统一缩减的定量空间。

**审稿期只读复算（尚未进入论文，不是新增前瞻实验）**：原rows SHA `550304d4c363a99f1c9d404375b2c9af15ad50351379614cad3aaf02ef818a79`。对每个model×scale×ref的64条RGB记录，令A=mean(a)、B=mean(b)，最优统一缩减lambda=clip(A/B,0,1)，风险=mean(reference_mse)−2lambda A+lambda² B。根agent与独立审稿agent分别重算一致，MSE恒等式最大残差1.09e−19。模型权重/输入/输出未改变，优化明确使用本测试集真值。

|参照|模型/倍率|后验最优lambda|相对native平均MSE改善|
|---|---|---:|---:|
|zero|LTE×2|1|0%|
|zero|LTE×4|1|0%|
|zero|SC×2|.993064|.001081%|
|zero|SC×4|1|0%|
|h3|LTE×2|.987788|.001438%|
|h3|LTE×4|.680949|.085304%|
|h3|SC×2|.960187|.013846%|
|h3|SC×4|.890775|.007361%|

这些是固定样本/固定分组/0≤lambda≤1统一输出插值族的精确经验最优，不是泛化置信界或部署效果，不排除逐图选择、lambda>1、改变方向或其他纠错。比只报告lambda=.5更能回答统一缩减的空间；需完整曲线及代价/救回比例呈现后再写主稿。

投稿前最小闭合项：①把主问题由解释完整图像原因收紧到失败类型、损害规模和统一幅度修正的能力；②复用已有输出形成完整lambda风险/救回/新增损害曲线与后验最优固定系数，标明真值依赖；③同一批保存输出作raw/[0,1]截断和固定裁边敏感性，呈现实质损失分布、按明确规则选取的自然图误差图；④清楚定义SC-INR及cell进入模型的位置，精简运行/审计琐节、旧预测支线。无需新网络、匹配训练或新任务；数值审计已通过不等于新颖性或录用已确认。当前判断为有投稿打磨价值的经验分析稿，尚非稳录用判断。


## 2026-10-18：自然图非零参照与跨种子补证，主线收束完成

导师要求以单一、扎实的故事完成硕士阶段分析型论文；本轮实际补实验，不做匹配训练。主论点收敛为：**原生cell相对zero平均有益，但具体图像/频带可受损；失败可分为过冲与方向错，且分类依赖明确的修正参照；局部救回不能代替整图及总体收益评估。** 归一化降为比较基础，旧失败预测器压缩为决策边界，受控输入路径/相位保留为条件性证据。当前4页技术＋1页参考文献PDF SHA `c94877fba93097ca85793dea335c6c24b68f2166f2ae59879a292ae7c6ba5056`，旧完整五页工程在 `versions/pre-cell-reference-20261018`，本轮版本在 `versions/cell-reference-5page`。未提交。

**新证据。** [完整协议、源码、审查与原件](../补证-20261018/cell-reference-v2/)；DIV2K valid0837–0868固定32图，区别于旧校准0001–0032、旧evaluation0825–0836及checkpoint验证0801–0810。LTE/SC-INR各继承seed1/seed2的epoch1000，×2/4、中心HR128、bicubic-AA LR、shave4、raw RGB及3非DC Chebyshev频带。固定权重/输入/query/target，比较relative cell=0/.5/2⁄3/1；native在×2为1，×4为.5。后三值都在实际训练cell数值支持内，但固定目标改变cell不是联合分布内样本，也不是关闭读出的等价消融。主对照为zero和固定h3(2⁄3)，另两个非native对照完整保留。

全256文件、1024 RGB数组先SHA冻结后读truth评分，3072行完整、无正式失败。每个下表格子包含32图×2种子=64整图记录，不是64独立图；image bootstrap共10000次，倍率分开，跨种子/倍率共同重采样图像。

|模型/倍率|native相对zero平均ΔPSNR及95%图簇区间|zero负整图|h3负整图（损失>.01dB）|half修正相对native总体MSE：zero / h3|
|---|---|---:|---:|---:|
|LTE ×2|+.912 [.593,1.292] dB|1/64|13/64（6）|+7.549% / +2.292%|
|LTE ×4|+.444 [.266,.669] dB|2/64|35/64（10）|+2.769% / −.058%|
|SC ×2|+.687 [.371,1.063] dB|10/64|17/64（9）|+5.463% / +1.836%|
|SC ×4|+.028 [.008,.053] dB|11/64|25/64（7）|+.264% / +.087%|

zero四组half代价及h3×2两组的95%区间都为正；h3×4两组区间跨0，不能说所有参照统一减半都伤总体。h3×4最差整图损失LTE .202dB、SC .136dB，多数负例幅度很小，负例数与实质损失分开。

**重要修正。** SC×2 zero的10条整图负记录全部在h3负集合内；SC×4 zero的11条与h3的25条完全不相交。这两个参照在×4时位于native两侧，不能把标签改变包装为新机制，也不能将“非零参照有负例”偷换成“同图负例跨参照不变”。zero整图负例跨seed交/并LTE1/2、SC10/11；h3为19/29与17/25。频带交/并另为LTE5/21、SC23/46（zero），不拿整图稳定性代替频带稳定性。新cohort也不支持把旧开发的“SC整图负损失99.52%来自过冲”写成模型固有属性。旧数据原件及其限定结果全部保留。

**验证与成本。** source `03c6e3ef`、protocol `7ed4105a`、独立预审 `41951503`；整棵冻结继承源码来自10月11日snapshot，未使用dirty SC-ASISR新实现。正式CPU1 nice10 GPU0，预定1800秒绝对/累计预算内637.272秒（准备2.376、推理630.063、评分4.742），没有训练。旧0801 preflight v1 8组native分块与整批输出差0，但汇总JSON的NumPy整数保存失败，27.283秒及FAILED原样保留；v2只修序列化和预先建议的统计汇总，在新根复用原预测验证评分.169秒，未重推理或改正式cohort。

独立审查脚本不导入实验代码，使用复数FFT/Parseval重算全量，117081项一次PASS，6.883秒CPU1 nice10；32PNG预处理0差，指标最大差8.79e−14、汇总3.41e−13、RGB-band-DC 3.42e−18，全部源码/权重/输入/输出hash及预算核通过。原件tar SHA `510fbf02318343cb9c1b9af1573d6a869ffbf71ca23293ccea460e65b40799ee`已完整下载，root本地401文件hash再次通过；审查JSON SHA `5fedf39ff0f8717c4f39adc1b067f3fea17ad221603f760530cb326f79cf5166`。评分审计不是网络独立重实现。

主表改为新cohort全部主对照，主图展示32图×两种子全部256条成对RGB记录，没有挑最好样本。旧合成曲线仍保留于figures/icassp-controlled.pdf及历史稿，正文保留完整生成参数/关键计数/效应量。独立整稿定向评审通过后，将摘要与结论的平均改善明确限定为relative to zero。完整编译/字体/全页渲染见编译核查.json。

**当前判断。** 支撑一篇范围明确的经验分析论文，强项是对照清楚、失败与代价可复现；未形成新架构、可靠自动修复规则或自然图纹理因果占比。下一步是导师审本单主线基线，无须为维持研究扩大训练。旧automation保持PAUSED。


## 历史主稿：原因诊断完整入稿（10月14日）

导师要求消除主稿落后于研究证据的问题。本轮已重构现有 main.tex / sec/icassp.tex，标题为 **When Does Cell Conditioning Harm Reconstruction? Diagnostics for Arbitrary-Scale Super-Resolution**。现行 main.pdf 为4页技术内容＋1页参考文献，SHA `ae4b614e5bfbd857df8e7eb80cb3cb1c9cad9f73b0ee6f3f4f47104daecc3f0c`。旧五页完整工程（PDF f818c4c9）保存在 versions/pre-cause-integration-5page；更早23页与各阶段归档不变。以下10月12日及以前条目是历史记录，所述“主稿保持”指当时状态。

主线收敛为：**先建立跨倍率比较坐标，再区分同倍率/同频带的读出失败类型，最后用固定模型受控输入路径和相位实验检验谱能量遗漏的条件。** 未将经典恒等式或相位原理包装为新理论；没有将单图预测器成功设为分析贡献的必要条件。

|正文位置|已纳入证据|保留的关键边界|
|---|---|---|
|§2|27组Nyquist归一化及逐图/权重诊断|原严格坍缩失败；均值不等于逐图规律；权重依赖HR|
|§3.1|G1/L2频带排序、简单band基线与两类消融区别|排序不证明积分读出；固定权重与重训不同|
|§3.2、表1|32旧cal图的过冲/方向错；频带与RGB不同；λ=.5局部救回与总体MSE变差|HR辅助输出混合，不是cell衰减算法；分母和负损失子集分别陈述|
|§4、图1|完整240条件的高频路径/相位控制；全部16相关phase-position曲线/面板|η伴随LR统计变化；8组非独立；η0同输入不能称相位适配；实际PSNR效应很小|
|§5|旧12图预测与32图内部迁移负结果、成本、限制|旧测试无事后调参；内部迁移仅开发OOF；未建立可靠指导|

图1来自未修改的stage-b原始 rows.json，制图代码为 figures/build-cause-figure.py，生成vector PDF并嵌入TrueType字体；数据来源/输出hash见 figures/cause-figure-sources.json。没有新实验或匹配训练。补入[Oppenheim–Lim 1981原文](https://dsp-group.mit.edu/wp-content/uploads/2024/11/ImportancePhaseSignals_1981.pdf)与IEEE元信息，明确相位重要性为经典先行；LIIF/IPE/Thera/HyRA等既有相关工作继续保留。

独立证据复核对原始Stage B行重新汇总，并核Stage A、transfer、源/协议/结果SHA；独立整稿科学审阅及定向复核均完成。三处必修已处理：自然图自适应机器精度边界与合成固定阈值分离；AUC只平均eligible cells（LTE3/6、SC5/6）且bootstrap固定OOF不重拟合；2.04秒为单CPU线程墙钟累计而非process CPU秒。另明确严格救回G(.5)>1e-8、六mode幅值及proxy误差符号。

服务器已有TeX/BibTeX编译完成，本地与服务器正文/图/PDF SHA一致；5页最终130dpi逐页审阅，无截断/重叠、无LaTeX warning/overfull/未解析引用，Underfull4保留；13字体全部嵌入，无Type3。完整记录见编译核查.json。当前可审阅源码包为 main-source.zip；旧 icassp-target-source.zip 仍是历史10月11日版本。最终冻结工程在 versions/cause-diagnostics-5page。

当前结论是条件性失败诊断，而非已揭示自然图完整因果机制或获得实用增益。作者栏依导师指示暂留内部评审标记，未提交；不启动匹配训练，旧automation保持PAUSED。下一节点为导师审阅本整合稿，再决定自然纹理/另一已有checkpoint的针对性验证是否值得投入。

## 三步受控补证：过冲、方向错和高频路径/相位效应（10月12日）

导师要求由更换预测指标转为原因区分，并在真实证据上灵活收敛主张；旧部署gate不是论文价值的唯一标准。原件[readout-cause-v1](../补证-20261012/readout-cause-v1/)，没有匹配训练/新网络/GPU。当时五页PDF f818c4c9保持前版；该补证现已纳入10月14日现行稿，本节保留原实验记录。

**第一步：保存输出的失败分型。** Stage-a协议e9327b4b、分析db1fb78f；32旧校准图0001–0032，LTE/SC ×2/4、全RGB和3原频带。d=Fc−F0，a=<t−F0,d>，b=||d||²，Gλ=2λa−λ²b。原负例中a<0为错误方向，0<a<b/2为过冲，近数值边界单列。该平方误差恒等式和HRoracle最优λ是经典工具，输出插值不等于cell干预，也不是可部署预测器。全部正负例和λ=.25/.5/.75/1直接像素评分保留。

|32校准图的描述性结果|LTE|SC-INR|
|---|---:|---:|
|负频带：过冲/方向错|12/1（合13）|14/13（合27）|
|方向错占负频带损失|8.91%|74.57%|
|全RGB负case：过冲/方向错|1/0|7/2|
|过冲占全RGB负损失|100%|99.52%|
|半修正救回负频带|11/13|6/27|
|半修正整体RGB MSE相对native|+2.405%|+.966%|

分母是图像×倍率×频带记录，不是独立图数。band负损失与RGB负损失基于不同子集，不混称“SC主要方向错”。冻结后描述cross-band-descriptive.json未改原summary：LTE10个含负频带case中9个fullRGB正、1负；SC21个中12正、9负。三频带不含DC，full gain−Σband gain的DC残项单列，可以确认同case补偿存在。

**第二/三步：固定模型的高频路径与相位干预。** Stage-b协议a25b0ac0、源码c1129f95，独立静态审查后执行排除cohort预检。HR64/LR16固定×4，均值.5、每mode幅值.04，低分支(2,0)/(0,2)/(2,2)固定；四高分支模板（两个中频、两个高频）和一个低频sham，1组全零相位+3组固定随机相位、4预定平移。所有逐频幅值/目标能量不变，平移作为嵌套位置对照，不能当独立相位机制。

周期bicubicAA由三周期基底→中心LR16取得，另核每场景直接二维插值。target x=.5+xL+xH在η0/.5/1间不变，y=A(.5+xL)+ηA(xH)；低频sham始终完整A x。η改变指定高频输入路径，也改变LR能量/梯度/统计，不能称剥离这些因素后的纯alias效应。观测/预测均不clip。全240条件保留，只148唯一输入、296组成对输出；四模板全部η0相位共享4个平移输入，不增加独立样本量。输入/真值先冻结，网络仅读观测，全输出SHA冻结后评分；生成式固定格点不是自然未见盲test。

|中心32 ROI、中频band b2，η0→1|LTE|SC-INR|
|---|---:|---:|
|读出gain下降|29/32|32/32|
|负收益数变化|18→32|15→31|
|gain正→负|14/32|16/32|
|alignment正→负|14/32|16/32|

上述不是全RGB：中心全RGB的中频模板gain正→负为LTE21/32、SC14/32；alignment正→负为26/32、15/32。全64 ROI同中频band gain下降均31/32，负数LTE14→31、SC15→23，支持方向趋势，精确数量依赖边界。高频模板η效应正负都有，不能称混叠总有害。所指为native相对zero-cell的读出增益，不能混写成η干预后的总重建质量改善/恶化。

在η1、固定模板与位置内只换phase，两高频模板×四位置每模型8/8组在目标高频band出现正负收益，全64 ROI同band亦8/8。逐频幅谱两ROI最大差3.22e−13（未归一化FFT量）。结论是**目标功率谱单独不足以确定该固定模型的读出收益**，不等于LR功率谱/梯度相同，不排除更丰富观测统计。η0相位间输入/预测相同而truth不同，评分差只能表述为未观测目标结构下的风险变化，不能称网络适配到了相位。

效应量小：高频η1中心全RGB，LTE native相对zero-cell ΔPSNR约−.00730至+.01236dB，SC约−.00164至+.000891dB；属于诊断证据，不包装成大幅性能收益。合成、周期边界、单checkpoint、zero-cell可能训练外等范围保留；未识别自然图的因果占比或网络内部机制。

**验证及原件。** Stage A独立review.json f4ff3cb5：24,244项通过，64truth/128pred/512row/26汇总/2048次λ核算，像素恒等式最大差4.20e−18。分析含import1.522663秒+审查.519829秒=2.042492秒，CPU1 nice10，0新推理。B正式240条件/80family/3840评分、1280η配对、960phase统计、960交互全部保留；shell85.32秒，launch至完成85.067222秒，CPU84.871502秒，RSS约626MiB，预检3.59秒；900秒绝对/600累计及root02:15UTC截止未延长。全程GPU/SR训练0。

B独立audit.py681c0154、audit.json924524ea：340,449项一次通过，wall4.718176秒/CPU4.423937秒；完整构造/逐频幅谱/η共享/sham/像素/配对/交互/汇总/hash复算。Gλ、配对与交互误差约1e−18量级，λ等派生row最大绝对差2.17e−11；sham差0。root另以独立NumPy Keys a=−.5构造周期bicubic矩阵，与torch逐值0差，M SHA f5f69de8；两checkpoint、全部输入/预测及源码再次核SHA。评分审查不是网络重实现。

完整tar SHA51ad0194cd94acb4d5948a4cae86959918c4c4079974887e68901ce02a3b5782已下载。completion.json与stage-b/interpretation.json含完整状态/成本/边界。图[失败分型](../补证-20261012/readout-cause-v1/figures/failure-modes.png)及[全相位/位置路径曲线](../补证-20261012/readout-cause-v1/figures/controlled-alias-phase.png)来自保存数据，不挑最优相位；后者每面板16相关条件细线及有限格点均值，不是置信区间。

可收敛的表述：**频带排序掩盖不同读出失败类型；过冲可局部减弱，错误方向不能靠正向缩小修复；指定输入路径与目标结构能改变收益，故目标谱量和全局衰减不足以给出统一解释。** 几何关系本身经典，贡献落在受控证据和范围，不以统一预测成功或新定理包装。本轮未启动匹配训练，也未从合成符号翻转直接扩自然cohort。

## 最新开发关口：下层自重建收益不能可靠迁移为目标收益（10月12日）

沿同倍率、同频带图像差异主线，完成[readout-internal-transfer-v1](../补证-20261012/readout-internal-transfer-v1/)。明确假设：将观测LR按同倍率再次bicubic-AA缩小，用冻结native/zero-cell模型重建回已知LR，所得可观测signed gain能预测原目标gain；同时检验将它单独加入既有norm/spectrum ridge是否有增量。该内部跨尺度思路由经典internal recurrence启发，不称新理论：[Glasner/Bagon/Irani2009作者页](https://www.wisdom.weizmann.ac.il/~vision/SingleImageSR.html)、[ZSSR CVPR2018官方摘要](https://openaccess.thecvf.com/content_cvpr_2018/html/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.html)。本轮只读上述原始页面；CVF PDF访问403，未声称核全文。ZSSR训练单图网络，本轮仅使用已有checkpoint诊断，不训练SR模型。

协议4e426659、源码33e4ee42在推理前冻结，独立设计/源码审查原件保留。32张旧校准train0001–0032，LTE/SC各一个epoch1000冻结checkpoint，×2/4，共128组成对下层预测；原128组成对HR输出复用，无重复原推理。下层目标64/32、输入32/8，裁边2/1对应原HR128 shave4的相同空间比例；raw RGB、不clamp输出、三个固定Chebyshev频带，只有低层输入插值后clamp。两层cell数值及FFT离散支持不同，非完全相同物理频率/神经响应。全部512下层特征和128预测SHA cohort5c63422b冻结后才读原校准标签；原标签SHAbc6db431与旧冻结一致，旧test0825–0836未读。

8折按整图留出，所有倍率/频带共同留出，训练折28图、验证折4图；标准化/median只在训练折，ridge固定lambda1。七控制与两候选均保留，不搜索参数/阈值/层数。该OOF只隔离选择器拟合，图像来自原网络训练划分，checkpoint具体入训身份未逐一重审，不能称网络未见图验证。预定三重门槛：分类召回/精度/FPR与全部基线比较、同cell宏AUC较norm改善、RGB策略较全部基线改善；要求同候选/模型共同成立。本轮0/4通过。

|模型/方法|负例识别|正例误判|合并BA|同倍率同频带宏AUC|RGB MSE较始终native|
|---|---:|---:|---:|---:|---:|
|LTE 幅度/频谱ridge|5/13|46/175|.561|.601|+.112%|
|LTE 下层直接迁移|5/13|27/175|.615|.747|+.239%|
|LTE 加下层特征ridge|5/13|45/175|.564|.599|+.091%|
|SC 幅度/频谱ridge|9/27|44/152|.522|.697|+.009%|
|SC 下层直接迁移|12/27|29/152|.627|.694|+.142%|
|SC 加下层特征ridge|10/27|44/152|.540|.678|+.006%|

分母为图像×倍率×频带记录；LTE13负例分布9张图、SC27负例分布18张图，非独立记录。每模型192频带记录，LTE4ties、SC13ties保留但从sign/AUC剔除；RGB每模型64case。AUC只在有正负例的固定cell内计算，再宏平均：LTE3/6，SC5/6。下层直接迁移AUC相对norm差的图像簇95%区间分别[-.06153,.36494]与[-.12550,.11997]。2000固定OOF预测重采样中分别1253/1309有效，缺任一固定cell类别的747/691次整体列missing，不以剩余cell替代。区间未重拟合OOF选择器，只是条件描述；不声称联合95%或外部泛化保证。OOF band-median因训练折标签不同可产生cell内得分变化，该排序不代表引入了图像内容信息。

下层有益而目标有害LTE8/13、SC15/27；SC×4高频8个负例中7个如此。直接迁移并未稳定学得同cell图像排序。它在SC合并BA上的改善不足以建立实际指导：RGB MSE为.00248615096，始终native为.00248262080，norm策略.00248284487；相对norm改善95%区间[-7.304e-6,-4.791e-7]。LTE相应.00249272208、.00248676822、.00248955445。对始终native，两模型直接迁移MSE均更差，平均PSNR分别-.01596/-.01052dB。LTE关闭6case，其中误关5个受益case；SC关闭13case，误关7个受益case，并保留4个有害case。完整逐cell/每图/九策略/所有门槛见summary与audit，不选择性只报BA改善。

完整程序27.974503秒（CPU27.851583秒、RSS约640MiB），外部bash time总29.68秒；CPU1 nice10、GPU0、SR训练0；预检shell2.73秒，独立两轮审查wall2.050219秒/CPU1.99453秒，root128下层预处理核查.098010秒。旧原目标推理成本仍见原阶段resources，不当作部署免费。初始/usr/bin/time不存在导致Python前启动失败，日志保留；改已有bash time-p后源码未变。预先1800秒正式绝对/900秒累计、root19:06:11UTC截止预算均未延长。

原件tar SHA70f8e65368d85130c973afb0ebfe693fbacc6aebe22f1e49c96f07a54b923a3c，已完整下载及验证。独立audit-v2.py SHA09a97072，audit-v2.json SHA677b709e，21,657项通过、最大数值差3.55e−15：全像素/FFT/512 OOF/2000bootstrap/门槛独立NumPy重算；不是模型重实现。首次audit.json保留193本地权限失败，原因是Python安全tar data filter将本地副本0444变0644；tar及现场remote193原件全0444且SHA一致，v2改核冻结归档原权限，不改数值判据。root另核两checkpoint实际字节SHA及128下层输入同torch预处理0差；同原语重算不称独立插值实现。见completion.json完整验证范围和失败记录。

研究判断：本轮否定的是固定条件下直接/线性增广的下层收益迁移能可靠指导目标读出，未证明所有内部图像方法不可能，也未识别造成符号翻转的原因。再缩小引起内容/混叠变化只是待检验解释。按预定停止条件不扩大此候选、不读取新test、不启动匹配训练。现行五页主稿PDF f818c4c9保持不变；这里保留开发负结果及接续边界，不将其另升级为确认性主贡献。


## 当前主线补证：平均一致性与逐图一致性的区别（10月12日）

回到已授权的ASISR细节恢复/尺度读出主线，完成有界回顾性分解 ../补证-20261011/recovery-residual-v1/；不重训、不推理，不将经典权重/协方差恒等式包装为新理论。协议58b036ac、分析8024ff69、结果201c9e42，四输入先冻结；27完整组、666图像×模型×种子组合、567pair/bin，全部图像和失败保留，原D逐值复现。固定跨倍率权重只作HR诊断，不替代原primary estimand。

|数据集（各9组）|原pool D范围|固定权重D范围|固定后降低组数|逐图D≤.05（每组）|
|---|---:|---:|---:|---:|
|Set14|.049285–.061595|.022732–.029419|9/9|7–9/14|
|BSD100|.011165–.015397|.008103–.012546|9/9|23–26/30|
|Urban100|.014695–.020178|.029299–.053793|0/9|14–17/30|

Urban100全部原pool组通过，但固定后LTE seed1/3分别.053793/.052043超过.05。SC seed1在u=[1.25,1.5]、倍率2→4（结果按2减4）的pool差+.02935363，逐图变化分量+.09237812与权重变化分量−.06302449相互抵消；16图为正、14图为负。分量是精确描述，不能解释为独立物理原因；D本身不作加法分解。全部四类最大pair/bin差均位于u≥1，未扩区间/选图，不证明所有误差均集中那里。归一化658/666逐图改善，8个Set14例外留存；逐图.05仅沿用原group阈值作描述，不是新预注册单图标准。

独立audit-v2.py SHA7dee31bd：80,243项通过，维度无关最大绝对差1.066e−14，原FFT能量最大相对差4.742e−16；所有27×2000配对image bootstrap、所有LOO、全分量/权重/ESS独立重算。原audit.py/.json初次752原能量abs1e−11失败保留（1e7量级能量的加法舍入）；v2只对原能量使用abs1e−11+rtol1e−12，维度无关量原abs1e−11未变，不能称原审查通过。分析.340842秒+两轮审查6.421880秒=6.762722秒，CPU1 nice10、GPU/推理/训练0；服务器编译/制图/编排不计入CPU分析时长。root末核全input/source/result/audit SHA通过，见completion.json；无原RGB/模型重实现。

新近primary-source覆盖已写相关工作：
- [Breaking Scale Anchoring](https://arxiv.org/html/2512.05132v2)，Wang/Zhang，v2 2026-02-08。§5 Eq7已有Nyquist-normalized编码，AppendixD Eq25设定归一化gain跨尺度近似一致，Eq26/27归一化PSD及加权误差。其任务为时空预测并改变训练编码/损失；本工作固定ASISR输出的误差分析不同，但归一化坐标本身非首创，不继承其假设为我们的定律。
- [Frequency-response analysis of deep super-resolution models and visualization of degradation mismatch](https://www.jstage.jst.go.jp/article/nolta/17/4/17_2026NCP0010/_html/-char/en)，Kawai等，NOLTA IEICE17(4):1418–1433，2026，DOI10.1587/nolta.2026NCP0010。§3方向正弦探针/投影及窗口/half-pixel，§4退化失配；频率响应探针与失配诊断已有先行，不称首创。

当前五页PDF SHA f818c4c963565a5c06616b32415e02921edd298d0302005ca765bb098243f34c；服务器paper-recovery-residual-v1 TeX/BibTeX、五页140dpi全审阅与residuals.png核查通过，warning/overfull/未解析引用0，Underfull4、13嵌入字体。原五页工程/PDF（7e3b85f9）完整存versions/pre-recovery-residual-5page，旧23页证据稿仍保留。当前冻结源码归档见versions/recovery-residual-5page；旧icassp-target-5page ZIP仍是历史版本。

进一步升华的准确范围：归一化改善比较坐标，却不自动建立逐图恢复定律；pool权重可放大也可掩盖变化，均图也存在不同图像正负抵消。该诊断未证明读出损害的原因或得到新预测器，既有负预测门槛失败不能借此修复。下一研究关口是可观测内容/修正特征能否超出band/scale和magnitude/spectrum强控制；任何改动预测须另冻结开发/校准/评价，不把本轮HR权重引入观察限定指导，不用已见测试事后调参证明泛化。保持未提交、不做匹配训练、旧automation PAUSED。


## 前一ICASSP五页版本（10月11日归档）

**10月11日导师授权的ICASSP整稿阶段完成，未提交。**现行主稿为 *Frequency Recovery and the Limits of Readout Explanations in Arbitrary-Scale Super-Resolution*。核心为“Nyquist归一化的描述性稳健性，以及粗频带排序不足以建立所测试的读出解释或逐图决策”；只保留两项经验贡献。移出E3条件线性比较、Gaussian选择、EEG、开发控制和长审计，不依赖审稿补充材料闭合核心论证。旧23页完整证据工程/PDF仍在versions/negative-readout-23page（SHA1d7f3a53），旧结果与失败不改。

新增有界离线稳健性原件：../补证-20261011/icassp-robustness-v1/。协议30d38b71、分析45f0bc6d、结果d0e0012d；87输入先hash冻结，7551项独立核验及root输入/源码/结果hash复核通过。分析＋独立审查累计6.56秒，CPU1 nice10、GPU0、训练0、推理0。24固定设定×27组均保留Nyquist归一化改善；全域原池化严格.05通过19/27、均图27/27，seed1为6/9与9/9，不能以新均图修复原失败。G1粗序34/36（池化）与36/36（均图）均保留，footprint仍与band index同秩。SC负RGB数在shave0/4/8为6/5/5，各24；原频带负数16/14/14，替代边界10–16。LTE RGB0，负频带1–5仍少。冻结RGB政策未重拟合，各裁边SCproxy改善区间仍跨0、directional略差，LTEproxy有害。这是已见输出的描述性敏感性，不是新确认性test。

相关工作直接核LIIF/IPE已有cell损害与干预、Thera已有Gaussian/bicubic失配、HyRA细粒度频率诊断；新增Magid纹理错误分析和DIV2K/Set14/BSD/Urban数据来源。不存在“首次发现cell有害”或积分首创主张。两轮独立科学审查已处理：L1不同bin支持、G1 .5/L2 .7、L2同口径footprint comparator、单seed冻结干预、LTE snapshot缺原clamp、原模型validation使用未知、完整分类AND RGBgate；评分复核不称模型重实现。

主稿PDF SHA=7e3b85f951aa87ca2a4fd40755349b895449571ba09a2ffad85c44af7c4d01a1。现有服务器paper-icassp-target-v1编译通过，本地/远端源码与PDFhash一致；140dpi五页全审阅、4页正文/第5页仅参考文献，无warning/overfull/未解析引用，Underfull4保留，13字体均嵌入。模板官方spconf/IEEEbib来自 https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/ ，正文含实际Claude/Codex使用披露。完整核查见编译核查.json；作者信息按导师指示后填，不作研究阻塞。

投稿时点：2027常规截止2026-09-23已过。此为ICASSP贡献/技术目标稿，非已提交或录用；未来会次规则发布后再核，未臆造supp审稿通道。完成后判断：经验主线清楚、证据可信，适合按ICASSP目标由导师评审；仍缺超过强简单基线的可靠负收益预测与广泛独立推广，未证实A类竞争力，不启动匹配训练、扩大test调参或恢复旧automation。

## 历史23页证据稿：负收益预测结果完整入稿

**10月11日前一阶段（23页证据稿）：负收益预测检验完整结束。**固定32校准图与12评价图、原LTE/SC权重，全部预测/推荐先冻结后评分；104输入、208预测、512校准/192测试/128alias评分与完整成本独立审查通过。SC14个负收益频带：proxy识别4个，directional ridge识别6个，与norm/spectrum基线相同却多误杀2个正收益（17 vs15）；BA .568低于基线.585。LTE仅1负例，不能证明稳定负预测。两个候选×两个模型综合门槛0/4；SCproxy pooledMSE小幅改善但区间跨零、平均PSNR略降。27/64固定同观测对收益符号反转，仅界定固定模型/任务，非通用不可能性。正式单CPU nice10 GPU0 SR训练0，224.192秒；不启动匹配训练、不按test结果扩参数/图，旧定时安排仍PAUSED。当前23页唯一论文已写完整方法/结果/失败及限制，旧21页工程保留。来源沿用论文SOURCES，原件位于补证-20261011/negative-collected。

PDF SHA256=1d7f3a53638e06df26f9e495a2834b35a6dfd848d71b5a56720df411772eafcf；正文/参考文献1–8，补充9–23；最终TeX/BibTeX与23页全渲染/17–19页150dpi审阅通过，warning/overfull/未解析引用0，Underfull 18项保留。旧21页完整源码/PDF位于versions/pre-negative-readout-21page。新增C.1节和表19/20，正文5.5与局限同步收窄；GSURE和range/null-space相关工作核primarysource，公式/代理/ridge为经典工具，不包装首创。

|键|直接原件与验证|
|---|---|
|NEGATIVE-READOUT-20261011|../补证-20261011/negative-readout-v3/protocol.json SHA b0b1c892、experiment.py SHA e073961f；v1/v2/v3及失败根保留。校准0001–0032、评价0825–0836，仅本次接续新增评分；不称全球未见。forecastcohort 23bf225189366b9a25664f73dbba9e7aadbfeee5491e4b0892347919fad57e12 先于score冻结。完整208预测和193input/truth SHA核通过。|
|NEGATIVE-AUDIT-20261011|../补证-20261011/negative-readout-v2/audit.py、v3/audit.json；独立numpy-only重算512/192/128全部行、岭/预测/统计/10k簇bootstrap/gates/cost，1.791秒，forecast/testscore最大差4.44e-16、统计1.11e-16；alias Fourier独立1.33e-15。未独立重跑模型、原PNG预处理或torch bicubic值；审查自身初次实现/FFTdtype失败原件留档，正式根未改。|
|NEGATIVE-MANUSCRIPT-20261011|../补证-20261011/negative-readout-v3/manuscript-review.json独立从全部行复算两表及PSNR、alias例/限制，科学/数字措辞通过；编译核查.json保留最终PDF/source SHA及全页QA。|
|NEGATIVE-ARCHIVE-20261011|../补证-20261011/negative-collected/codex-20261011-negative-complete-v1.tar.gz SHA25992815f3944516c2fb44bab22d38822a1db7af42a03f6137103589394ec598，本地/远端一致。依赖原snapshot/两个checkpoint沿用旧collected完整归档；新归档不复制训练环境。|

正式完整224.192秒，阶段5.785/213.014/3.350/.704秒，5,865,472RGB查询、CPUuser/system215.174/8.830秒、RSS1534.1MiB、CPU1 nice10 GPU0 SR训练0。v1预检5.056秒、v3预检7.048秒；v2首case前CPU SIG11根和元数据原位保留，缺精确资源记录，120秒固定上限，不记成零成本或宣称已知全部历史费用。正式900秒绝对/累计预算内完成、守护6652/6653已退出。

SCproxy负召回4/14、precision4/19、BA.51355；directional6/14、6/23、BA.56773，norm基线6/14、6/21、.58498，方向BA差区间[-.16667,.14286]。proxy RGBMSE .00412906 vs native .00414218，但改善区间[-6.81e-6,4.78e-5]跨0，meanPSNR-.004669dB；directional MSE .00414470。LTE1负例1图、3562/10k bootstrap缺负类；不强调高BA。全部四门槛失败，不能以此支持匹配训练。27/64alias对符号反转（9LTE、18SC），相关固定对照不是总体比例；观测一致误差2.44e-15、低真值proxy误差4.86e-8、仅task/weights具体反例。

primarysources：[Eldar GSURE](https://arxiv.org/abs/0804.3010)，[AAAI 2023 Null-space SR](https://ojs.aaai.org/index.php/AAAI/article/view/25372)。不将自然bicubic前向与合成box前向混同。


## 历史18页论文基线（2026-10-08，现行稿见顶部）

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

## 2026-10-11：负收益预测检验的历史启动记录（已完成，现状见顶部）

**10月11日新增当前节点：**导师已授权围绕“何时读出修正损害重建”推进。固定32图0001–0032校准、12图0825–0836前瞻评价，128中心裁块×2/4，原LTE/SC epoch1000冻结权重。用观测bicubic参考的修正方向与变化能量预测raw gain，对照band/scale median、energy-only、范数/谱ridge；全部test预测/系数/推荐先SHA冻结再评分。另16个alias共享观测对检验代理边界。源码e073961f、协议b0b1c892，新增目录negative-readout-v3位于补证-20261011。v1预检5.056秒通过；v2首case输出前CPU SIG11保留；v3仅CPU1+faulthandler，7.048秒预检通过。正式根codex-20261011-negative-formal-v3，守护初始PID6652，固定900秒绝对/累计预算、nice10 GPU0 SR训练0，以现场状态为准。尚未读取正式评分，不预判有效；不做匹配训练，旧automation仍PAUSED。

协议../补证-20261011/negative-readout-v3/protocol.json；静态独立审查../补证-20261011/negative-readout-v2/review.json。候选为zero-shot proxy和directional ridge，后一候选与norm ridge仅差一个alignment特征；经典校准工具，不称新无偏风险理论。12图是本次接续新增评分cohort，不称全球未见。完整负召回、precision、FPR、BA与fullRGB关闭修正policy分别核门槛；10k图像簇paired bootstrap不是联合95%保证，两候选逐一报告，不挑testwinner。准备阶段HR用于生成LR；forecast阶段只读观测及校准truth。native+zero两份推理成本计入，不称pre-inference指导。正式原件完成后独立核全部inputtruth/hash/forecastfreeze/校准与评分及cost。
