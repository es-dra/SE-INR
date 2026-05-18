# SC-ASISR

这是 arbitrary-scale image super-resolution（ASISR）的科研工作区。当前主线是
`SC-INR`：用 sampling-consistent Fourier implicit decoding 改善尺度外推，而不是
宣称严格数学意义的 scale equivariance。

## 当前研究问题

LTE 会把输出 cell/scale 直接输入 learned phase branch，容易把采样尺度变成纹理相位
快捷通道。`SC-INR` 的核心约束是：

- 图像内容产生 `coef(z)`、`omega(z)` 和可选的 `phi(z)`；
- 输出 footprint 通过解析 `sinc(omega * cell / 2)` 进入 observation response；
- 禁止 learned `phase(cell)` 这类直接由尺度移动纹理相位的捷径。

## 主入口

| 路径 | 用途 |
| --- | --- |
| `entrypoints/train.py` | 训练入口 |
| `entrypoints/eval_full.py` | 离散尺度 benchmark 入口 |
| `scripts/analysis/build_canonical_benchmarks.py` | 生成论文主 benchmark 表 |
| `scripts/analysis/model_registry.py` | 结果 key 与论文展示名映射 |
| `scripts/viz/prepare_qualitative_figure.py` | 定性图导出 |
| `configs/registry/models.yaml` | 模型、checkpoint、raw key 的命名注册表 |
| `configs/registry/protocols.yaml` | 评估协议注册表 |
| `paper/ARTIFACTS_ALLOWED.md` | 论文可引用 artifact 白名单 |
| `paper/claims_evidence_matrix.md` | claim 与证据边界 |
| `docs/project/current_state.md` | 当前状态恢复入口 |

## 常用命令

```bash
# 训练最终候选 SC-INR
python entrypoints/train.py \
  --config configs/train-div2k/train-sc-inr.yaml \
  --name sc-inr --saveFolder artifacts/checkpoints/seed1 \
  --device 1 --seed 1

# 离散尺度 benchmark
python entrypoints/eval_full.py \
  --device 1 \
  --save_root artifacts/checkpoints/seed1 \
  --models SC-INR \
  --output artifacts/raw_results/seed1/benchmark_sc_inr_manual.json \
  --skip_existing

# 重建 canonical benchmark 表
python scripts/analysis/build_canonical_benchmarks.py
```

## 结果入口

| 类型 | canonical 路径 | 说明 |
| --- | --- | --- |
| checkpoint | `artifacts/checkpoints/` | 只保留模型权重、训练配置和训练日志；根目录 `save` 是 seed1 兼容入口 |
| raw benchmark | `artifacts/raw_results/` | 正式原始评估 JSON |
| 主 benchmark 表 | `artifacts/derived/benchmarks/` | 论文主表优先使用这里 |
| 辅助诊断 | `artifacts/derived/diagnostics/` | consistency、response、机制诊断等 |
| 证据索引 | `artifacts/derived/evidence/README_zh.md` | 辅助证据总入口 |
| 定性图 | `artifacts/derived/paper_figures/qualitative_selected_seed1/` | 用户确认的候选图 |
| 历史留档 | `artifacts/legacy/` | 仅作 provenance/audit，不是当前证据入口 |

## 命名边界

- `SC-INR`：最终候选，checkpoint 使用 `artifacts/checkpoints/seed*/sc-inr`。
- `SC-INR-NoPhi`：旧 no-phase 主线，checkpoint 使用 `sc-inr-nophi`。
- `SC-INR-NoPhi-Signed`、`SC-INR-NoSinc`、`SC-INR-FixedOmega`：消融模型。
- `LIIF`、`LTE`、`LIIF-EQ`、`LTE-EQ`：对比模型。
- `SC-INR-EQ`：探索性 Rot-E 结合，不是主方法。

旧 raw key 或历史名如 `SC-INR+PhiZ`、`SC-INR-Adaptive`、`sc-inr-phiz` 只作为
raw-result provenance 出现；`save`/`artifacts/checkpoints` 下不再保留旧命名模型目录。
新命令、新文档和论文正文应使用 `configs/registry/models.yaml` 里的 canonical 名称。

## 兼容路径

根目录只保留少量兼容 symlink：

- `models` -> `src/models`
- `datasets` -> `src/datasets`
- `utils.py` -> `src/utils.py`
- `save` -> `artifacts/checkpoints/seed1`
- `save-seeds` -> `artifacts/checkpoints/seeds`
- `results` -> `artifacts/results`
- `logs` -> `artifacts/logs`
- `Data` -> `artifacts/data_local`

`results` 只放指向 canonical benchmark JSON 的便捷链接；raw JSON 在
`artifacts/raw_results/`，主表和全模型指标在 `artifacts/derived/benchmarks/`。
新脚本和新文档应优先使用 canonical 路径；兼容路径只用于旧命令恢复。

## 不能过度宣称

当前证据支持：

- `SC-INR` 在当前 3 seed protocol 下相对 LIIF/LTE 有小幅 OOD/ALL PSNR 正增益；
- decoder-side sampling consistency / scale-decoupled observation 是合理表述；
- seed1 小样本 footprint oracle 诊断显示 `SC-INR` 比 `SC-INR-NoSinc` 更跟随
  HR box-average proxy，但这仍是 diagnostic；
- consistency 指标只能作为诊断，不能单独证明 observation modeling 正确。

当前证据不支持：

- `SC-INR` 严格 scale equivariant；
- `SC-INR-EQ` 是主方法；
- sinc 是唯一因果因素；
- 单 seed 诊断结果升级为主表结论。
