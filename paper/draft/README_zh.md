# SC-INR 冻结内部技术记录

这不是活跃投稿稿。P1.12 的 IPE-LIIF/Thera 双 baseline 硬门已失败，SC-INR 新方法 claim 已停止。
本目录仅保留已经实现的结构、公式、实验数字和证据边界，不能以“补齐相邻实验即可恢复”为前提
继续扩写。正式收口见 `docs/project/sc_inr_route_closeout_2026-08-07.md`。

## 文件

- `main.tex`：冻结的英文内部技术记录。
- `references.bib`：当前只包含 LIIF 和 LTE 的基础引用。
- `figures/`：从 qualitative candidate pool 复制出的两张用户确认图。
- `main.pdf`：由冻结记录编译的内部审阅 PDF，不是投稿成稿。

## 编译命令

```bash
cd paper/draft
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## 证据边界

- `SC-INR` 的三 seed 结果只支持当前 bicubic 协议下的小幅 OOD/ALL 正 mean delta，实现事实可保留。
- `SC-INR-NoPhi` 的 cross-grid consistency、NoSinc 和 cell intervention 都是 diagnostic，不能证明
  sampling correctness、sinc 因果或最终 RGB exact box integration。
- 草稿不得改写为 strict scale equivariance、SOTA、稳定方法优势或可投稿的新方法贡献。
- IPE-LIIF/Thera 外部 provenance 若改变，只允许重开来源与公平性审计，不自动恢复本草稿。
