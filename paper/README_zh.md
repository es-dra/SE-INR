# 论文材料

当前分析论文为[When Does Cell Conditioning Harm Reconstruction?](current/main.pdf)，源码在[current/](current/)，证据脉络及所有正负结果见[current/SOURCES.md](current/SOURCES.md)和[研究现状](../docs/project/current_state.md)。这是对已有模型的尺度读出失败诊断，未投稿、未做匹配训练；不是恢复SC-INR新方法主张。

`draft/`仍为早期SC-INR冻结内部技术记录，`current/versions/`为分析论文历次版本，两者身份不同。下列历史约束仅针对旧新方法路线，不禁止已授权的当前分析工作。

---

# SC-INR 论文材料状态

`paper/` 只保留 SC-INR 的冻结内部技术记录、证据边界和历史复现映射，不是活跃投稿工作区。

2026-08-07 的 P1.12 审计未通过 IPE-LIIF/Thera 双 baseline 硬门，P1.13 未启动，SC-INR 新方法
claim 已停止。`draft/main.tex` 与 `draft/main.pdf` 保留是为了记录已经实现的结构、公式、实验数字和
结论边界；不得把它们当作可投稿稿件，也不得在没有新 Node Contract 的情况下继续补模型、表格或
投稿叙事。

## Canonical 入口

- 路线收口：`docs/project/sc_inr_route_closeout_2026-08-07.md`
- 当前状态：`docs/project/current_state.md`
- claim 边界：`paper/claims_evidence_matrix.md`
- 内部引用白名单：`paper/ARTIFACTS_ALLOWED.md`
- 历史模型名映射：`paper/model_taxonomy.md`
- 冻结草稿说明：`paper/draft/README_zh.md`

只有新的外部来源证据先满足 P1.12 恢复条件，并由项目负责人批准新的 Node Contract，才允许重新
审计；通过来源审计也不等于自动恢复 novelty、性能或投稿 claim。
