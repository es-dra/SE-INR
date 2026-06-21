# Cell Intervention Visualization

本目录用一个小型可视化实验展示 `SC-INR` 区别于 `LTE` 和 `SC-INR-NoSinc` 的核心行为。

## 协议

- 固定同一个 HR crop 生成 x4 LR 输入；
- 固定同一个 query grid；
- 只改变 decoder 输入的 cell multiplier：native `1` 和 large `4`；
- 候选 crop 先按 GT 高频/局部纹理强度预筛，再按预定义 tracking 指标排序；
- 模型：`LIIF,LTE,LTE-PhaseZ,SC-INR,SC-INR-NoSinc`。

## 主要产物

- `candidate_pool.csv`：GT 高频预筛候选 crop。
- `cell_intervention_metrics.csv`：逐 crop、逐模型指标。
- `selected_examples.csv`：按预定义分数排序后的候选。
- `figures/selected_*.png|pdf`：native cell、large cell、signed delta 与 oracle 对照图。
- `figures/selected_summary.png|pdf`：selected examples 上的平均指标柱状图。

## 当前最强展示例子

- top example: `urban100/img_002.png` crop `(x=192, y=192, size=96)`
- SC-INR tracking RMSE: `0.04227822`
- LTE tracking RMSE: `0.04551088`
- NoSinc tracking RMSE: `0.07090569`
- selection gate pass rate in candidate pool: `0.833`

## 如何解释

该实验最想展示的不是最终 PSNR，而是 cell path 的行为差异：

- `LTE`：cell 进入 learned phase，large-cell 改变可能表现为纹理相位/局部结构移动。
- `LTE-PhaseZ`：phase 来自 feature 而不是 cell，是去掉 cell-conditioned phase 的对照。
- `SC-INR-NoSinc`：去掉 analytic response 后，large-cell 输出几乎不变，说明它不是 footprint-aware。
- `SC-INR`：large-cell 产生非零响应，并且在 selected examples 上更接近 HR box-average oracle 的变化。

## 使用边界

这是 mechanism demonstration，不是 benchmark。它支持“SC-INR 的 cell path 行为区别于 LTE/NoSinc”，不能单独证明 `SC-INR` 全局视觉质量更好、sinc 是唯一因果，或最终 RGB 是 exact box integral。

## 复现命令

```bash
python scripts/analysis/visualize_cell_intervention.py \
  --device cpu \
  --datasets urban100,bsd100 \
  --max_images 3 \
  --crop_size 96 \
  --stride 96 \
  --max_crops_per_image 3 \
  --max_candidates 18 \
  --num_select 4 \
  --out artifacts/derived/diagnostics/incode_method_support_2026-06-20/cell_intervention
```
