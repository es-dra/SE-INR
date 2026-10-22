# Two coordinates of arbitrary-scale SR: recovery lives in u, scale conditioning acts in r

Original registered stages retain frozen protocols, predictions, scoring scripts and subsequent scoring amendments. Stages 15–16 are retrospective saved-data audits. Failed predictions are kept. Historical SERVER_PATH.txt files and RUNS.md locate original runs and their retained large-output hashes.

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
| | | P5 best cell within 10% of native, scales 2/3/4 | **fail** 20/27 satisfy; best fixed cell gains ≤0.017 dB |
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
| 13 standard benchmarks | matched arms, full Set5/Set14/BSD100/Urban100, Y-PSNR | descriptive | tinydec −0.24 to −0.28, no-cell −0.09 to −0.13 (all scales, both seeds), RDN +0.25 to +0.35, SwinIR +0.34 to +0.46 dB |
| 15 recovery/filter decomposition (2026-10-23) | existing DIV2K v11 aggregate statistics; saved Set14 float16 outputs, 2 seeds | retrospective, no new training or inference | for doubled cell, R and w jointly account for 41.4% / 41.5% of the measured ×2→×6 MSE-cost decrease; non-scalar net remainder accounts for 54.4% / 58.5% (DIV2K / Set14) |
| 16 residual structure and step controls (2026-10-23) | saved Set14 outputs; original float32 moments on 214 Set14/BSD100/Urban100 images | retrospective, no model calls | frequency/image localization and finer projections; matched-step controls contradict universal sensitivity loss; manuscript revised |

Transfer-function analysis (`01-cell-mismatch/transfer.py`) motivated the approximation `P_m ≈ w_m(r)·P_1`. The **2026-10-23 offline audit qualifies its scale-independence and explanatory claims**: the fitted strength w changes with scale even when the affected frequency range is similar. For doubled cell, the original proxy Q = Σ(1−w)²RE is 43.6%, 71.9%, 81.5%, and 250.1% of the measured DIV2K cost at ×2/3/4/6. Agreement at ×3/4 includes cancellation between omitted terms and does not establish an exact or prospective law. Small mismatch (m = 4/3) gives a positive proxy but a slightly negative measured cost at ×4/6; halved cell has only 9.7–19.7% proxy coverage.

The exact retained-band accounting is D = Q + G + H, where G = Σ2w(1−w)Re⟨T−P,P⟩ and H = Σ(||ε||²−2Re⟨T−P,ε⟩), ε = P_m−wP. A symmetric two-factor decomposition of ΔQ separates R and w while retaining ΔG and ΔH. For ×2→×6, the actual-cost shares [R,w,G,H] are [22.69,18.68,4.22,54.41]% on DIV2K and [27.26,14.27,−0.04,58.51]% on Set14. The Set14 H share has a paired image-bootstrap 95% interval of [41.99,66.96]%; intervals condition on the two existing trained seeds. Fitting w per image still leaves H at 53.71% of the endpoint decrease, so pooling images alone does not close the gap.

These are descriptive **RGB MSE** decompositions on 240-pixel crops, with the original 16 r bands (0<r<0.5, excluding DC and the exact Nyquist outer ring). They are not dB component shares or full-image benchmark claims. Set14 includes Parseval checks, the omitted-frequency audit, all-non-DC sensitivity, and 2,000 paired image bootstrap samples; DIV2K v11 has aggregate statistics only, so no image CI can be recovered for it. All old results remain unchanged. The analysis does not identify the training benefit of cell.

Reproducible inputs, definitions, and current results: [`analysis_spec.json`](15-energy-filter-decomposition/analysis_spec.json), [`analyze.py`](15-energy-filter-decomposition/analyze.py), [`final/results.json`](15-energy-filter-decomposition/final/results.json), [`figure`](15-energy-filter-decomposition/final/decomposition.png), and [`validation.json`](15-energy-filter-decomposition/validation.json). The new server output is `/workspace/SE-INR/runs/codex-20261023-energy-filter-v1`; only saved predictions were read (20.26 seconds CPU extraction, no model calls). Both offline audits are incorporated in the revised manuscript and PDF.


Stage 16 localizes the **net H decrease**: r ∈ [0.1875,0.375) contributes 70.9% on Set14 and 78.5% on DIV2K. Comic, monarch and zebra contribute 64.1% of the Set14 net decrease; barbara contributes −18.8% because doubled cell corrects substantial native error at ×2. Leave-one-image-out refits retain H shares of 56.8–61.4%. Net-cost allocations differ from rankings by residual energy.

The real-projection audit fits each saved image/seed separately: A per band, B per band/channel, C per frequency, D per frequency/channel. A→B→D and A→C→D are nested; B and C are not. At ×2/×6, A retains 93.2%/84.7% of pooled residual energy, B 93.1%/84.3%, C 45.5%/35.4%, D 40.7%/29.0%. Much of the original remainder reflects within-band frequency variation; remaining D energy is quadrature relative to native phase (negative gains absorb π flips). This is an in-sample representation diagnostic, not a causal mechanism or improved predictive model. D energy decreases 119.0→14.9 ×10⁻⁶ MSE, but its net residual cost changes +84.8→−1.9 ×10⁻⁶. Source hashes, zero/small-power audits and exact identities pass.

Native decoder cell is c = 2/s per axis, so actual finite step is h = 2(m−1)/s. Raw pooled residual energy falls 292.7→51.5 ×10⁻⁶ for m=2, but division by h² gives 292.7→463.3 ×10⁻⁶. On original float32 moments, matched positive steps at (s,m)=(2,1.3333),(6,2) give output-change energy ratios J₆/J₂ of 2.263 [1.306,3.973], 1.090 [0.909,1.278], 0.766 [0.613,0.959] on Set14/BSD100/Urban100. Paired image bootstrap uses 2,000 samples, fixed trained seeds. Negative matched steps differ: BSD100 response energy falls 26% while mean MSE cost rises 218%. Net cost is D = J−2Re⟨e,δ⟩; dB cost is 10 log10(1+D/V). Cross-scale features, query offsets and training support also differ; these comparisons do not isolate a Jacobian or causal scale effect.

Stage 16 source and final results: [specification](16-residual-structure/analysis_spec.json), [localization-v2](16-residual-structure/localization-v2/report.json), [projection-summary-v3](16-residual-structure/projection-summary-v3/summary.json), [step-controls-v1](16-residual-structure/step-controls-v1/report.json), [validation](16-residual-structure/validation.json), and [paper figures](16-residual-structure/paper_figures.py). Earlier intermediate directories are retained. The original tracked paper is archived in 16-residual-structure/inputs/paper-before-residual-audit.tar. Remaining limits are prediction on genuinely new cases, actionable design rules and the mechanism of cell's training benefit. No new training or inference was started.
