# 论文候选图入口

本目录保存可进入论文或周报的候选图。正式引用前仍以
`paper/ARTIFACTS_ALLOWED.md` 为准。

## 当前内容

- `qualitative_selected_seed1/`：用户确认的 Urban100 x8 定性例图；只能作为 selected
  examples，不能证明平均视觉质量。
- `lte_scinr_mechanism_2026-05-18/`：LTE vs SC-INR 机制对比图；只说明 cell
  进入 decoder 的结构差异，应配合 `artifacts/derived/diagnostics/lte_scinr_mechanism_2026-05-16/`
  的 seed1 机制诊断使用。

## 使用边界

候选图不能替代 benchmark、multi-seed 结果或机制诊断。涉及机制解释时，需要同时说明
seed、数据覆盖、proxy 属性和不能过度宣称的边界。
