# SE-INR：任意倍率超分的“两个坐标”

研究问题：任意倍率超分（ASISR）的重建收益从哪里来？为什么 ×4 以上改解码器、改尺度输入（cell）几乎不再带来提升？

当前主线（2026-10-20 起）：
- **恢复度只由 u 决定。** 网络能恢复多少，只取决于按 LR Nyquist 归一化的频率 u = 2·s·r，与解码器设计无关；训练和编码器只提高 u≥1 的部分。
- **cell 作用在 r 上。** cell 相当于在 HR 像素频率 r 上加的一个低通，它的杠杆受限于该频率上网络已恢复的能量，所以倍率越大，cell 越无关紧要。

全部证据、预注册预测与未通过项见 [studies/two-coordinates](studies/two-coordinates/README.md)，论文大纲见 [paper/OUTLINE.md](paper/OUTLINE.md)，研究历程见 [docs/HISTORY.md](docs/HISTORY.md)。

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/` | 模型（EDSR、RDN 编码器；LIIF、LTE、SC-INR 解码器）、数据加载、工具 |
| `entrypoints/` | 训练与评测入口 |
| `configs/` | 训练与测试配置；`*-200ep.yaml` 为匹配训练实验 |
| `studies/` | 每个研究阶段的冻结协议、代码、评分脚本和结果摘要 |
| `paper/` | 当前论文 |
| `docs/` | 研究历程 |

`datasets`、`models`、`utils.py` 是兼容链接：指向 `src/` 下对应模块。

## 运行

训练：`python entrypoints/train.py --config configs/train-div2k/train-liif-edsr-200ep.yaml --name <name> --device 0 --seed 1`

服务器 `research-175:/workspace/SE-INR` 的布局见该目录下的 `INDEX.md`。数据集和 checkpoint 不入库。

## 历史

2026-10-19 之前的全部内容（SC-INR 方法路线、Codex 的 cell 负收益诊断稿、各论文版本与原始记录）都完整保存在标签 [`archive-2026-10`](../../tree/archive-2026-10) 中，用 `git checkout archive-2026-10` 即可查看。
