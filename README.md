# SE-INR：连续图像表示与尺度读出诊断

当前研究 **目标像素大小（cell）如何影响任意倍率重建，以及何时尺度修正会损害结果**。沿用LTE和作者由LTE衍生的SC-INR已有模型，研究负收益的类型、受控原因和针对性干预；目前是ICASSP目标的分析稿，未投稿，不提出新网络或通用修复器。

## 阅读入口

1. [研究现状](docs/project/current_state.md)：主要结果、反例、边界，以及早期路线到当前问题的脉络。
2. [当前论文](paper/current/main.pdf)与[完整出处和阶段记录](paper/current/SOURCES.md)。
3. [实验原件索引](experiments/october2026/manifest.json)：协议、代码、完整评分、失败版本与原件包SHA；大型原件仍在科研服务器原目录。

最新受控干预：固定SC权重和×4，保持内部特征**总改变量**相同，只改变削弱分配，焦点16例中频负收益在正常/固定解码开关下均转正；完整32中频由31负/1正变为32正。焦点16例完整RGB误差下降中位**0.3996%**。但16例低频对照的**中心RGB**由正收益变负，说明这是特定失败的诊断证据，不能当通用方案。此前“输入使尺度权重漂移”的假设只救回1/16，未获得主要解释支持。

SC-INR是LTE衍生实验变体。早期新方法路线的[收口记录](docs/project/sc_inr_route_closeout_2026-08-07.md)、[冻结草稿](paper/draft/README_zh.md)、[模型命名](paper/model_taxonomy.md)与历史结果保留；当前分析论文不恢复旧性能/新方法主张。源码仍保留历史实现，模型/数据/协议以每项冻结记录为准。

服务器`research-175`：当前干净版本在`/workspace/SE-INR/SE-INR-current`；旧`SC-ASISR`工作树及全部未提交改动保留，`claude-*`/`codex-*`实验根、checkpoint与日志原位不动。读取进展从本页及研究现状开始，不把旧目录的训练计划当作新授权。

<details>
<summary>同步前服务器README（历史导航，原文保留）</summary>

# SC-ASISR / SC-INR

本项目研究任意倍率图像超分辨率。SC-ASISR 是项目目录名，SC-INR 是研究模型名；代码、训练配方和实验结果分开管理。

## 从这里开始

- [当前研究与运行入口](docs/project/current_state.md)：正在回答的问题、当前执行状态和主线材料。
- [维护清单](MANIFEST.md)：目录职责、命名和历史材料的处理方式。
- [模型实现](src/models/) · [训练与评价入口](entrypoints/) · [配置](configs/) · [测试](tests/)。

当前主线是 P／S／R 在共同初始化和数据条件下的读取方式比较。准备工作、正式训练和科学结论分别记录；是否启动全量训练以负责人决定和当前运行状态为准。

## 目录

| 目录 | 内容 |
| --- | --- |
| src/ | 可复用模型、数据加载与工具代码 |
| entrypoints/ | 训练、评价等运行入口 |
| configs/ | 配方、协议和模型注册 |
| scripts/ | 分析、评价与产物生成脚本 |
| tests/ | 数学性质、接口和恢复检查 |
| outputs/research-week-20260906/ | 主任务的执行状态入口 |
| artifacts/ | 数据、检查点、原始结果和必要复现快照 |
| docs/project/ | 当前入口与尚待归并的历史研究文档 |
| memory/ | 历史工作记录；不作为最新运行状态 |
| paper/ | 论文材料及其证据范围；不等于当前已接受结论 |

旧路线和旧结论按发生时间保留，不再占据首页的“当前状态”。大批历史合同、修订脚本和快照仍待在本轮活跃实验收口后按引用关系归并；不能仅凭文件旧、名称含 v2 或内容相同就删除。

本次入口整理前的原文保存在 [恢复包](artifacts/legacy/maintenance-20260911/navigation-before.zip)，包含原 README、维护清单和完整的历史状态页。

外层 /workspace/SE-INR/Data 是共享数据；models、datasets、save 等兼容链接不是重复副本。此轮整理未改变这些路径。

</details>
