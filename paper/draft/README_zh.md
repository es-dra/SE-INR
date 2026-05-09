# SC-INR 论文草稿

这是当前证据基础上的最小可编译论文草稿，不是最终投稿稿。

## 文件

- `main.tex`：论文草稿。
- `references.bib`：当前只包含 LIIF 和 LTE 的基础引用。
- `figures/`：从 qualitative candidate pool 复制出的两张用户确认图。
- `main.pdf`：已编译 PDF。

## 编译命令

```bash
cd paper/draft
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## 证据边界

- `SC-INR-NoPhi` 的 3-seed OOD 和 consistency 证据较稳。
- 最终候选 `SC-INR` 目前仍是 seed1 preliminary。
- 草稿中不得改写为 strict scale equivariance、SOTA 或最终多 seed winner。
