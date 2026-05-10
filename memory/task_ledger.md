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

- 状态：running / initializing datasets。
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

### 后续 gate

- 训练中：定期检查 PID、GPU、最新 log epoch。
- 训练完成后：运行 full benchmark，核心比较 `LIIF`, `LTE`, `SC-INR-NoPhi`,
  `SC-INR`；再更新 multi-seed summary 和 claim ledger。
