# Task Ledger

本文件记录长期运行任务的恢复信息。它不是论文结论，也不替代 daily memory。

## 2026-05-10 SC-INR seed2/seed3 Training

### 自动推进协议

- 目标：启动 final-candidate `SC-INR` 的 seed2/seed3 训练，补齐 multi-seed
  统计所需 checkpoint。
- 允许改动范围：项目恢复入口、任务账本、seed2/seed3 `sc-inr` checkpoint 目录、
  训练日志、daily memory；后台训练期间可推进 qualitative/table/diagnostic
  文档工作。
- 禁止事项：不覆盖 seed1；不删除或重命名已有 checkpoint；不改变训练协议；
  不把未完成 seed2/3 写成多 seed 结论；不启动额外新方法分支。
- 资源预算：使用两张当前空闲物理 GPU；命令必须显式设置
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<physical_gpu>`。
- 配置：`configs/train-div2k/train-sc-inr.yaml`。
- checkpoint rule：训练输出到 `artifacts/checkpoints/seed{2,3}/sc-inr/`，
  使用该目录内 `epoch-best.pth` 作为后续 benchmark checkpoint。
- 成功节点：两个训练均完成到 `epoch 1000/1000`，各自产生 `epoch-best.pth`、
  `epoch-last.pth`、`config.yaml`、`log.txt`。
- 最低有效进展：训练启动后，`log.txt` 记录 seed、CUDA_VISIBLE_DEVICES、
  dataset shape、首个 epoch train/val。
- 暂停条件：GPU 被他人占用；输出目录已有 `sc-inr` 正式 checkpoint 且无法判断是否可续；
  loss/val 出现 NaN/inf；进程异常退出；需要改变训练配置或公平性协议。

### 运行状态

- 状态：completed; full benchmark completed and summarized。
- seed2 tmux：`scinr_seed2_20260510`
- seed3 tmux：`scinr_seed3_20260510`
- seed2 command：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python entrypoints/train.py --config configs/train-div2k/train-sc-inr.yaml --name sc-inr --saveFolder artifacts/checkpoints/seed2 --seed 2`
- seed3 command：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python entrypoints/train.py --config configs/train-div2k/train-sc-inr.yaml --name sc-inr --saveFolder artifacts/checkpoints/seed3 --seed 3`
- seed2 PID：`64956`
- seed3 PID：`64959`
- seed2 log：`artifacts/checkpoints/seed2/sc-inr/log.txt`
- seed3 log：`artifacts/checkpoints/seed3/sc-inr/log.txt`
- seed2 stdout：`artifacts/checkpoints/seed2/sc-inr/stdout.log`
- seed3 stdout：`artifacts/checkpoints/seed3/sc-inr/stdout.log`

### 最近检查

- `2026-05-10 05:30 UTC`: 两个进程仍在运行，日志已记录 seed 与
  `CUDA_VISIBLE_DEVICES`；仍处于 in-memory dataset 初始化阶段，GPU 2/3 尚未开始
  明显占用。
- `2026-05-10 05:36 UTC`: 两个训练均进入正式 epoch。
  - seed2: `epoch 1/1000, train: loss=0.0379, val: psnr=28.9507`,
    ETA about `12.8h`。
  - seed3: `epoch 1/1000, train: loss=0.0382, val: psnr=29.2282`,
    ETA about `13.8h`。
  - GPU 2/3 memory about `2790 MiB` each, utilization about `90%/94%`。
- `2026-05-10 05:39 UTC`: 两个训练均到 `epoch 4/1000`。
  - seed2 latest: `epoch 4/1000, train: loss=0.0342, val: psnr=29.6217`,
    ETA about `12.2h`。
  - seed3 latest: `epoch 4/1000, train: loss=0.0343, val: psnr=29.1810`,
    ETA about `12.3h`。
  - GPU 2/3 utilization about `95%/95%`。
- `2026-05-10 05:41 UTC`: 两个训练均到 `epoch 6/1000`。
  - seed2 latest: `epoch 6/1000, train: loss=0.0338, val: psnr=29.5979`,
    ETA about `12.0h`。
  - seed3 latest: `epoch 6/1000, train: loss=0.0338, val: psnr=29.2746`,
    ETA about `12.1h`。
  - `epoch-best.pth` 和 `epoch-last.pth` 已在两个输出目录中产生。
- `2026-05-10 05:52 UTC`: 根据用户指出的工作流缺口，已将“工作流优化必须写入全局规则、
  不能只写项目产物”的规则追加到 `/home/admin/.codex/AGENTS.md`。
  - seed2 latest: `epoch 23/1000, train: loss=0.0331, val: psnr=29.8818`,
    ETA about `11.8h`。
  - seed3 latest: `epoch 24/1000, train: loss=0.0331, val: psnr=29.9114`,
    ETA about `11.8h`。
  - `save-seeds/seed{2,3}/sc-inr/log.txt` 经 symlink 指向
    `artifacts/checkpoints/seed{2,3}/sc-inr/log.txt`。
- `2026-05-10 06:31 UTC`: 发表进展审查时复核两个训练仍正常运行。
  - seed2 latest observed: `epoch 79/1000, train: loss=0.0321, val: psnr=29.8327`,
    log ETA about `11.8h`。
  - seed3 latest observed: `epoch 80/1000, train: loss=0.0321, val: psnr=30.0551`,
    log ETA about `11.7h`。
  - GPU 2/3 memory about `2790 MiB` each, utilization about `94%/93%`。
  - `epoch-last.pth` for both seeds updated at `2026-05-10 06:31`; no `nan`/`inf`、
    `Traceback` or `error` matched in current logs.
- `2026-05-10 06:51 UTC`: 使用新的 subagent token 高效协议做一次训练进展审查测试。
  - seed2 latest observed: `epoch 107/1000, train: loss=0.0320, val: psnr=29.7461`,
    log ETA about `11.8h`。
  - seed3 latest observed: `epoch 107/1000, train: loss=0.0316, val: psnr=29.9454`,
    log ETA about `11.7h`。
  - GPU 2/3 memory about `2790 MiB` each, utilization about `95%/95%`。
  - 主流程和只读 subagent 均未发现 `nan`/`inf`、`Traceback` 或 `error`；当前无触发暂停条件。
- `2026-05-10 06:55 UTC`: 回归主线状态诊断；监督 subagent 只读确认主线阻塞仍是
  final `SC-INR` seed2/seed3 完成后 full benchmark，主流程补 live checks。
  - seed2 latest observed: `epoch 113/1000, train: loss=0.0314, val: psnr=29.9524`,
    log ETA about `11.8h`。
  - seed3 latest observed: `epoch 114/1000, train: loss=0.0321, val: psnr=29.7367`,
    log ETA about `11.7h`。
  - GPU 2/3 memory about `2790 MiB` each, utilization about `95%/94%`。
  - `epoch-last.pth` for both seeds updated at `2026-05-10 06:55`; seed3 `epoch-best.pth`
    also updated at `2026-05-10 06:55`。
  - 当前日志未匹配到 `nan`/`inf`、`Traceback`、`error`、`exception` 或 OOM；无触发暂停条件。
- `2026-05-11 UTC`: seed2/seed3 训练均完成到 `epoch 1000/1000`。
  - seed2 final log: `epoch 1000/1000, train: loss=0.0305, val: psnr=30.0257`。
  - seed3 final log: `epoch 1000/1000, train: loss=0.0308, val: psnr=29.8925`。
  - 两个目录均有 `epoch-best.pth`、`epoch-last.pth`、`epoch-1000.pth`。
  - 已补 full benchmark，未覆盖旧 ambiguous raw JSON：
    `artifacts/raw_results/seed2/benchmark_sc_inr.json` 和
    `artifacts/raw_results/seed3/benchmark_sc_inr.json`，各覆盖 36/36，无缺失或 `None`。
  - 生成 3-seed final `SC-INR` 汇总：
    `artifacts/derived/analysis/benchmark_progress_2026-05-11/`。
  - 关键结论：final `SC-INR` 3-seed OOD `22.7578 ± 0.0316`，ALL
    `25.5298 ± 0.0403`；paired delta vs LTE OOD `+0.0504 ± 0.0417`，
    ALL `+0.0320 ± 0.0555`；paired delta vs `SC-INR-NoPhi` 基本为 0。
  - 后续口径修正：论文展示主表优先比较 final `SC-INR` 与 `LIIF`、`LTE`；
    `SC-INR` vs `SC-INR-NoPhi` 只展示 seed1 结构增量，不展示 seed2/seed3 两者差异。

### 后续 gate

- 本节点已完成。后续只需在论文制表时引用
  `artifacts/derived/analysis/benchmark_progress_2026-05-11/`，并避免把
  feature phase 写成稳定 PSNR 增益。

## 2026-05-10 SC-INR-EQ Exploratory Training

### 自动推进协议

- 触发依据：用户确认当前有多余 GPU，可在不等待 final `SC-INR` seed2/seed3 完成的情况下
  并行启动探索性训练。
- 目标：启动一个 `SC-INR-EQ` 单 seed exploratory 分支，检验 “SC-INR decoder contract +
  Rot-E plumbing” 是否能正常训练。
- 允许改动范围：新增 `sc_inr_eq` 模型、训练配置、smoke gate 脚本、task ledger 和
  daily memory；训练输出隔离到 `artifacts/checkpoints/seed1/sc-inr-eq/`。
- 禁止事项：不占用 GPU2/3，不改 final `SC-INR` seed2/seed3 配置或 checkpoint；不把
  exploratory 单 seed 写成主线结论；不把 Rot-E 收益归因于 analytic sinc；不照搬
  `LTE-EQ` 的 cell-conditioned phase。
- 模型 contract：`coef(z)`、`omega(z)`、`phi(z)` 只由 feature 生成；cell/scale 只通过
  analytic sinc response 进入。
- 配置：`configs/train-div2k/train-sc-inr-eq.yaml`。
- 成功节点：smoke gate 通过；训练进入正式 epoch；最终完成后生成 `epoch-best.pth`、
  `epoch-last.pth`、`config.yaml`、`log.txt`。
- 暂停条件：smoke gate 失败；loss/val 出现 NaN/inf；训练进程异常退出；GPU0 被更高优先级
  任务占用；需要改变公平性协议或论文主结论。

### 实现与验证 gate

- 新增文件：
  - `src/models/sc_inr_eq.py`
  - `configs/train-div2k/train-sc-inr-eq.yaml`
  - `scripts/analysis/smoke_check_sc_inr_eq.py`
- 修改文件：
  - `src/models/__init__.py`
- 修正过的问题：首次 smoke 发现 `Fconv_PCA(kernel_size=1)` 会在 phase head 中产生 NaN；
  已改为 `Fconv_1X1`，并为 no-bias zero tensor 注册 buffer，避免 `.to(device)` 后设备不一致。
- 通过的 gate：
  - `python -m py_compile src/models/sc_inr_eq.py scripts/analysis/smoke_check_sc_inr_eq.py`
  - YAML parse for `configs/train-div2k/train-sc-inr-eq.yaml`
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/analysis/smoke_check_sc_inr_eq.py --device cuda:0`
- smoke 结果摘要：
  - omega init max abs error `7.450581e-09`
  - forward/backward output shape `(1, 256, 3)`，loss finite
  - scale cells x2/x4/x8/x16/x30 finite
  - response cell delta `3.166991e-01`

### 运行状态

- 状态：running / training normally。
- tmux：`scinr_eq_seed1_20260510`
- command：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python entrypoints/train.py --config configs/train-div2k/train-sc-inr-eq.yaml --name sc-inr-eq --saveFolder artifacts/checkpoints/seed1 --seed 1`
- output dir：`artifacts/checkpoints/seed1/sc-inr-eq/`
- log：`artifacts/checkpoints/seed1/sc-inr-eq/log.txt`
- stdout：`artifacts/checkpoints/seed1/sc-inr-eq/stdout.log`

### 最近检查

- `2026-05-10 16:06 UTC`: 训练进程仍在运行，日志已写出 `seed: 1`、
  `CUDA_VISIBLE_DEVICES: 0`；尚未写出首个 epoch。GPU0 memory about `2625 MiB`，
  utilization `0%`，判断仍处于初始化/缓存数据阶段。final `SC-INR` seed2/seed3 仍在
  GPU2/3 运行，未被新任务中断。
- `2026-05-10 16:10 UTC`: `SC-INR-EQ` 进程仍在运行但尚未写出首个 epoch；GPU0 memory
  about `2127 MiB`，utilization `0%`，CPU high。该启动时间已偏长，暂判断为
  in-memory dataset 初始化或 I/O/CPU 争用风险；未见 traceback/OOM，暂不停止。
- `2026-05-10 16:16 UTC`: 进程仍在运行，尚未写出首个 epoch；RSS about `10.97GB`，
  CPU high，GPU0 memory about `1001 MiB`。已核对 `image-folder` 的 `cache: in_memory`
  会在 dataset 构造阶段同步读入图片；当前更像数据缓存阶段偏慢，而不是模型计算失败。
  若继续长时间无日志，应考虑是否临时改为 `cache: none` 或等待当前缓存完成。
- `2026-05-11 UTC`: 训练正常进入中段；最新检查到 `epoch 326/1000,
  train: loss=0.0311, val: psnr=30.2522`，无 NaN/inf/traceback。
  tmux 仅剩 `scinr_eq_seed1_20260510`；GPU0 memory about `2879 MiB`。
  独立只读审查认为实现满足 contract：`coef/omega/phi` 由 feature 生成，cell
  只进入 analytic sinc response，没有 LTE-EQ learned `phase(cell)`；但旋转符号、
  通道 flatten 顺序、`eff_x/eff_y` 和 axis-aligned cell sinc 语义仍需属性测试证明。
  复跑 smoke gate 通过：
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python scripts/analysis/smoke_check_sc_inr_eq.py --device cuda:0`。
- `2026-05-11 UTC`: 修复 `artifacts/checkpoints/seed1/sc-inr-eq/log.txt`
  被截断的问题。症状是 `log.txt` 只从 epoch 318 开始，缺失 seed/dataset/model
  header 和 epoch 1-317；`stdout.log` 仍保留完整打印记录。已从 `stdout.log`
  恢复干净日志并原子替换 `log.txt`，保留原截断版备份：
  `log.txt.truncated_backup_20260511_034100.bak`。修复后检查到 epoch 1-334
  连续、无缺失、无重复，训练进程继续正常追加。

### 后续 gate

- 继续训练，但只作为 exploratory 分支保留；不要提前进入论文主 claim。
- 训练完成或中途出现异常后，再决定是否做 full benchmark。
- 后续必须补属性测试：cell 变化不得改变 `coef/omega/phi`，只改变 response；
  `phase_map` 对不同 cell 不变；旋转输入时 transform 通道按预期循环置换；
  analytic sinc 与直接 cell 平均积分对齐。
