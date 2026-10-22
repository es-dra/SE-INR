# Two coordinates of arbitrary-scale SR: recovery lives in u, scale conditioning acts in r

Every stage froze its protocol, predictions and scoring script (hashes in each `FROZEN.sha256`) before results were read. Failed predictions are kept. Large per-bin rows stay on the server (`LARGE_ROWS.sha256`); each folder's `SERVER_PATH.txt` gives the full run root on `research-175`.

Notation: `s` = scale, `r` = HR frequency (cycles/HR pixel, Chebyshev radius), `u = 2·s·r` = frequency over the LR Nyquist, `R(u) = 1 − err/target energy` per band, `m` = cell multiplier (cell = m × native).

| Stage | Data / models | Frozen prediction | Outcome |
| --- | --- | --- | --- |
| 00 u-profile re-analysis | saved DIV2K 0837–0868 outputs (LTE/SC, seeds 1–2) | exploratory | zero-cell contrasts are out-of-support artefacts; in-support cell cost sits mostly at u ≥ 1 |
| 01 cell mismatch | Set14/BSD100/Urban100; LIIF/LTE/SC × seeds 1–2; ×2/3/4/6; m ∈ [0.5, 2] | H1 ≥80% of cost at u≥1 in ≥90% of cases | **fail** 75% (median share 0.88) |
| | | H2 smaller cell costs more | **fail, reversed**: larger cell costs more (33/144) |
| 02 two coordinates | fresh DIV2K 0869–0900; seeds 1–3 | P1 R(u) collapses across scales (≤0.05) | **fail** 45/54 (×2 edge bin deviates) |
| | | P2 decoders equal (≤0.03) | pass 18/18 (max 0.010) |
| | | P3 cost of m=2 peaks at r∈[0.19,0.375) | pass 36/36 (all at r≈0.25) |
| | | P4 cost ~ recovered energy (ρ≥0.7) | pass ρ = 0.77–0.78 |
| | | P5 native cell optimal | **fail** 20/27; best fixed cell gains ≤0.017 dB |
| | | P6 cell leverage decays with scale | pass 9/9 (≈2.4 → 0.4 → 0.13 → 0 dB) |
| 03 other decoder families | GaussianSR (EDSR enc.), RDN-BTC (external) | Q1 GaussianSR R(u) equal to EDSR decoders | pass 18/18 |
| | | Q2 RDN-BTC higher R at u≥1 | **fail** 0/12; external checkpoint underperforms under this degradation |
| | | Q3 cell locus r≈0.25 | **fail** for both non-LIIF families (leverage still decays with scale) |
| 04 training prior | LIIF/LTE/SC epoch 100 vs 1000; official EDSR ×4 | R1 training raises only u≥1 | pass 12/12 above, 9/9 unchanged below u<0.75 |
| | | R2 official EDSR higher R | **fail** 0/4 (degradation mismatch, −0.8 dB) |
| | | R3 cell leverage grows with training | pass 3/3 (scoring bug fixed post hoc; conclusion unchanged) |
| 05 matched encoder | LIIF, same recipe, EDSR vs RDN, 200 epochs | T1–T3 RDN raises R only at u≥1; leverage decays | pass 12/12, 9/9, pass (+0.31/+0.23/+0.23/+0.17 dB) |
| 05b decoder budget | 2×64 decoder; no cell input | D1 small decoder ≤0.05 dB loss at ×4+ | **fail**: −0.14 to −0.20 dB at all scales, only u≥0.75 |
| | | D2 no-cell loss decays, ≤0.03 dB at ×4+ | **fail (narrow)**: −0.09/−0.03/−0.04/−0.00 dB |
| | | D3 encoder gain > decoder loss | pass (0.23 vs 0.15 dB at ×4) |
| 07 SwinIR encoder | LIIF + SwinIR vs EDSR at equal 50 epochs (eval crop fix, amendment 3) | S1–S3 | pass 12/12, 9/9, pass (+0.17/+0.18/+0.12/+0.07 dB); epoch 200: pass 12/12, 9/9, pass (+0.41/+0.32/+0.26/+0.25 dB) |
| 08 seed-2 replication | all matched arms, seed 2 | K1–K3 | pass 11/12; 12/12, 9/9; 12/12, 9/9 |
| 09 full images | Set14/BSD100/Urban100 full | F1 decoder invariance; F2 collapse; F3 bicubic gap | pass 54/54; **fail** 53/72; pass |
| 11 transfer replication | DIV2K, three decoders | W1–W3 | pass 9/9, 6/6, 105/105 |
| 12 Gaussian-blur degradation | EDSR/RDN/no-cell retrained | G1 gain bins match incomplete-linear bins; G2 capacity also acts below Nyquist; G3 RDN gain; G4 cell effect vanishes | pass 20/24; **fail** (gain only from u≈0.75; even EDSR recovers u<0.75 almost fully); pass (+0.33/+0.27/+0.23/+0.19 dB); pass |

Transfer-function analysis (`01-cell-mismatch/transfer.py`): the cell acts approximately as an HR-frequency low-pass on the native output, `P_m ≈ w_m(r)·P_1`, whose shape is nearly scale-independent. For m = 2, `(1−w)²|P_1|²` ranks per-band cost with Spearman 0.86–0.91 and accounts for 70–77% of it. For small mismatch (m = 4/3) the first-order cross-term dominates and the filter model does not hold.
