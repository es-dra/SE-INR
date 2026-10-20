# Third-party code

Parts of this repository are adapted from the projects below. Their licences continue to apply to the adapted files.

| Files | Source | Licence |
| --- | --- | --- |
| `src/models/liif.py`, `src/models/mlp.py`, `src/models/models.py`, `src/datasets/*` (except `gauss_wrapper.py`), `src/utils.py`, `entrypoints/train.py`, `entrypoints/test.py` | LIIF, Yinbo Chen et al. — https://github.com/yinboc/liif | BSD 3-Clause |
| `src/models/lte.py` | LTE, Jaewon Lee and Kyong Hwan Jin — https://github.com/jaewon-lee-b/lte | BSD 3-Clause |
| `src/models/swinir.py` | SwinIR, Jingyun Liang et al. — https://github.com/JingyunLiang/SwinIR, as packaged in the LTE repository | Apache 2.0 (SwinIR); BSD 3-Clause (LTE packaging) |
| `src/models/edsr.py` | EDSR-PyTorch, Sanghyun Son et al. — https://github.com/thstkdgus35/EDSR-PyTorch, via LIIF | MIT |
| `src/models/rdn.py` | RDN implementation as distributed with BTC (Byeonghyun Pak et al.) — https://github.com/ByeongHyunPak/btc, itself following LIIF | MIT |

`src/models/sc_inr_adaptive.py`, `src/datasets/gauss_wrapper.py`, `tools/` and `studies/` are original to this project.

Full licence texts are in [`licenses/`](licenses/).
