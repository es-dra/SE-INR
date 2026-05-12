# Raw Results

这里保存正式 raw metric JSON。派生表格和图应从这些文件生成。

## 当前文件

- `seed1/benchmark.json`：seed1 旧主 benchmark，不含最终候选 `SC-INR` 的 clean key。
- `seed1/benchmark_signed_phiz.json`：seed1 含 `SC-INR-NoPhi-Signed` 与最终候选
  `SC-INR` 的完整 benchmark；历史 raw key 分别为 `SC-INR-Signed` 与 `SC-INR+PhiZ`。
- `seed1/benchmark_sc_inr_nosinc.json`：seed1 NoSinc 消融 benchmark。
- `seed1/benchmark_sc_inr_eq.json`：seed1 exploratory `SC-INR-EQ` benchmark。
- `seed2/benchmark.json`：核心三模型 seed2 benchmark。
- `seed2/benchmark_sc_inr.json`：最终候选 `SC-INR` seed2 benchmark。
- `seed3/benchmark.json`：核心三模型 seed3 benchmark。
- `seed3/benchmark_sc_inr.json`：最终候选 `SC-INR` seed3 benchmark。

## 使用规则

- 论文主 benchmark 不直接读 raw key，应使用 `artifacts/derived/benchmarks/`。
- raw key 的历史歧义由 `configs/registry/models.yaml` 和
  `scripts/analysis/model_registry.py` 解释。
- 已删除旧 continuous/FCE/phase-intervention raw；它们不再是当前证据入口。
