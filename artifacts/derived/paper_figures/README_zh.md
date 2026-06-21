# 论文候选图入口

本目录保存可进入论文或周报的候选图。正式引用前仍以
`paper/ARTIFACTS_ALLOWED.md` 为准。

## 当前内容

- `qualitative_selected_seed1/`：用户确认的 Urban100 x8 定性例图；只能作为 selected
  examples，不能证明平均视觉质量。
- `liif_vs_inr_mechanism_2026-05-24/`：普通坐标 INR vs LIIF 类局部隐式表示机制图；
  只说明从全局坐标函数到局部特征条件化共享 decoder 的变量组织变化。
- `lte_scinr_mechanism_2026-05-18/`：LTE vs SC-INR 机制对比图；只说明 cell
  进入 decoder 的结构差异，应配合 `artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/`
  的 seed1 机制诊断使用。
- `footprint_response_mechanism_2026-05-31/`：SC-INR analytic footprint response
  机制图；解释 `W(omega,c)` 的引入动机、sinc 来源和 cell 从 phase shortcut 到
  observation response 的语义转移。

## 使用边界

候选图不能替代 benchmark、multi-seed 结果或机制诊断。涉及机制解释时，需要同时说明
seed、数据覆盖、proxy 属性和不能过度宣称的边界。
