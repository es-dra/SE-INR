# Review Advice Analysis for SC-INR / SE-INR

This note evaluates the supplied reviewer-style advice against the current local
code and seed-1 auxiliary results. It is intended as a working research memo, not
as camera-ready paper prose.

## Evidence Checked

- Auxiliary metrics:
  `results/analysis/seed1_aux_metrics_all8/quality_summary.csv`
  and `consistency_summary.csv`.
- Paper artifacts generated from those files:
  `results/analysis/seed1_aux_metrics_all8/paper/`.
- Model implementations:
  `models/lte.py`, `models/sc_inr_fixed.py`, and
  `models/sc_inr_adaptive.py`.
- Current multi-seed status:
  `save-seeds/seed2/*/log.txt`, `save-seeds/seed3/*/log.txt`, and
  `results/analysis/benchmark_seed_mean_std.csv`.

## Main Verdict

The advice is directionally correct: the paper should be positioned around
sampling-consistent / scale-decoupled observation, not strict scale equivariance.
The seed-1 auxiliary metrics now give useful support for this framing: the
reconstruction gains over LTE are small but stable, while same-LR cross-scale
observation consistency improves sharply for SC-INR.

The most important caveat is that the current evidence still does not isolate
the sinc response as the sole causal factor. SC-INR changes both the
cell injection mechanism and the local Fourier parameterization, so a w/o sinc
control remains the highest-priority missing ablation.

## What Is Already Supported

### Positioning

Supported. The current implementation does not prove or implement a full
scale-equivariant map. It constrains how the output cell enters the decoder.
The safest language is:

- scale-sampling consistency;
- sampling-consistent Fourier implicit decoding;
- scale-decoupled observation;
- improved OOD scale robustness.

Avoid claims such as strict scale equivariance or scale-equivariant ASISR unless
they are explicitly scoped to future extensions.

### Function-Observation Separation

Supported as the correct conceptual framing. In `models/lte.py`, cell enters
through:

```python
q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
```

In `models/sc_inr_adaptive.py`, cell enters through the analytic response:

```python
sinc_x = analytic_sinc(omega_x * c_x / 2)
sinc_y = analytic_sinc(omega_y * c_y / 2)
W = sinc_x * sinc_y
```

This directly supports the claim that SC-INR changes the role of cell from a
learned phase input to an observation response.

### More Metrics Beyond PSNR

Supported. The paper-ready seed-1 tables now show:

- BSD100 OOD PSNR: SC-INR is +0.056 dB over LTE.
- Urban100 OOD PSNR: SC-INR is +0.052 dB over LTE.
- BSD100 OOD SSIM-Y: +0.00191 over LTE.
- Urban100 OOD SSIM-Y: +0.00101 over LTE.
- BSD100 consistency PSNR: +8.61 dB over LTE.
- Urban100 consistency PSNR: +7.77 dB over LTE.
- BSD100 texture consistency RMSE: -0.00590 vs LTE.
- Urban100 texture consistency RMSE: -0.01047 vs LTE.

These numbers support a mechanism-oriented story better than a pure benchmark
story.

### LTE-NoCell / LTE-FeaturePhase Caveat

Strongly supported. LTE-NoCell and LTE-FeaturePhase obtain very high consistency
PSNR, around +25 to +26 dB over LTE, but this mostly reflects weakened cell
response. They should be presented as diagnostic controls, not as better
super-resolution models. The useful comparison is the joint tradeoff between
reconstruction quality and cross-scale consistency.

## What Needs Correction or Careful Wording

### Sinc Unit Consistency

The advice correctly flags this as important, but the current code is internally
consistent if the paper defines sinc as the normalized sinc:

```math
\operatorname{sinc}(x)=\frac{\sin(\pi x)}{\pi x}.
```

The decoder uses `cos(pi * omega * delta)` and `torch.sinc(omega * c / 2)`.
For a box average of `cos(pi * omega * x)`, the response is therefore
`sinc(omega * c / 2)` under this normalized-sinc convention. The paper must state
this convention explicitly. If the paper instead defines sinc as `sin(x)/x`, the
formula must include the corresponding `pi` factor.

### Adaptive Omega Parameterization

The advice is right that frequency constraints need analysis, but there is a
more specific current issue: `SCINRAdaptive` uses

```python
self.omega_map = F.softplus(self.omega_conv(self.feat))
```

on both x and y components. This forces every frequency component to be
non-negative and therefore weakens directional frequency expressiveness. The
fixed model uses log-polar frequencies with negative components when angles
fall in the second quadrant, but the adaptive model cannot produce those signs.

This does not invalidate the current results, but it is a real architectural
limitation. A later bounded signed parameterization, such as
`omega_max * tanh(g(z))` or a polar `(rho, theta)` form, is more defensible.

### Negative Sinc Response

The risk is real in theory, but with the current default `freq_max=2.0` and
relative output cell measured in LR feature coordinates, the OOD scales used
here often keep `omega * c / 2` near or below the first zero for the configured
frequency range. This should be checked empirically before claiming that sinc
side-lobes materially affect artifacts. A response-distribution diagnostic is
the right next step.

## Highest-Priority Missing Experiments

### 1. SC-INR w/o sinc

This is the most important ablation. It should keep the adaptive omega and MLP
structure but set the response to one:

```python
W = 1
```

or provide a config flag that disables sinc. This isolates whether the gain is
from analytic sampling response rather than the adaptive Fourier construction or
parameter count.

Expected paper use:

- If w/o sinc loses consistency and/or OOD quality, it directly supports the
  sampling-consistency mechanism.
- If w/o sinc matches SC-INR in PSNR but loses consistency, the paper
  should emphasize stability/consistency rather than reconstruction gain.
- If w/o sinc matches both, the sinc mechanism is not yet experimentally
  established and the method story must be revised.

### 2. Cell Response Curve

High priority. This would visually demonstrate the main mechanism: LTE can have
learned cell extrapolation, LTE-NoCell is almost invariant to cell, and SC-INR
should show smooth analytic attenuation. It is more directly tied to the method
than another benchmark table.

### 3. Frequency and Sinc Response Distributions

High priority because of the current `softplus` omega issue. Report:

- distribution of `omega_x`, `omega_y`, and magnitude;
- distribution of `omega * c / 2` for x4/x8/x16/x30;
- fraction of negative sinc responses;
- average attenuation by scale.

This can also tell us whether `freq_max=2.0` is too conservative or whether the
model is learning around the response.

### 4. Multi-Seed Mean and Std

Necessary for the final paper if PSNR gains remain near 0.05 dB. At the time of
this memo, `benchmark_seed_mean_std.csv` still only contains seed 1. Training is
in progress for additional seeds, but the final claim should wait for complete
mean/std.

### 5. Continuous Gain Curve

Useful and relatively low risk. The most paper-friendly form is
`PSNR_SC(s) - PSNR_LTE(s)` across scale. The current auxiliary table only covers
x4/x8/x16/x30; continuous curves already exist elsewhere in the repo and should
be regenerated/verified under the current protocol if used.

## Lower-Priority or Later Extensions

### SC-INR-EQ

This is scientifically interesting but should not preempt the w/o sinc and
response-curve experiments. The low-risk version
`edsr-eq-baseline + SC-INR decoder` is a good first step because it tests
orthogonality with Rot-E without forcing a full orientation-aware sinc design.

### LIIF+

This is useful for generality, but the minimal version should be treated as a
prototype. A defensible LIIF+ result would require a clean baseline/control:
no-cell LIIF plus a footprint-conditioned observation layer, ideally with a
fixed quadrature or integrated positional encoding. Do not overclaim
universality from a single LIIF+ prototype.

### Gaussian or Learned Response

Worth doing after w/o sinc. This separates the broader thesis
"analytic observation response helps" from the narrower claim
"box-filter sinc is best." If Gaussian performs similarly, the paper may need to
frame sinc as one principled instance rather than the unique solution.

## Recommended Paper Claim Strength

Safe:

> We decouple continuous local Fourier modeling from pixel-area observation and
> inject output cell size through an analytic sinc response, improving OOD scale
> robustness and same-LR cross-scale observation consistency.

Too strong for current evidence:

> We achieve scale equivariance.

Also too strong until w/o sinc:

> The PSNR gains are caused by the sinc response.

More defensible now:

> The current evidence shows that replacing learned cell-conditioned phase with
> a sampling-consistent response preserves or slightly improves OOD quality while
> substantially reducing cross-scale inconsistency.

