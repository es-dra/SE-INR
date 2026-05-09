# Raw Results

这里保存正式 raw metric JSON。派生表格和图应从这些文件生成。

## 当前文件

- `seed1/benchmark.json`：seed1 旧主 benchmark，8 模型。
- `seed1/benchmark_signed_phiz.json`：seed1 含 `SC-INR-NoPhi-Signed` 与最终候选 `SC-INR` 的完整 benchmark；历史 raw key 分别为 `SC-INR-Signed` 与 `SC-INR+PhiZ`。
- `seed2/benchmark.json`：核心三模型 seed2 benchmark。
- `seed3/benchmark.json`：核心三模型 seed3 benchmark。
- `continuous/seed1/*.json`：seed1 continuous-scale raw。
- `diagnostics/seed1/fce.json`：FCE 诊断 raw。
- `diagnostics/seed1/phase_intervention.json`：phase intervention raw。

## 注意

- `continuous/seed1/urban100.json` 中 EQ 模型覆盖不完整，不能作为完整 8 模型 Urban100 continuous 证据。
- `diagnostics/seed1/fce.json` 当前不是完整 FCE 全模型矩阵，引用时需说明覆盖。
