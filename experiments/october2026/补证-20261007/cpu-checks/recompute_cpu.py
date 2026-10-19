#!/usr/bin/env python3
"""Read-only D11 paired-source bootstrap correction and scalar-formula checks.

Original data and analyses are never modified. NumPy only; no GPU/inference.
"""
import argparse
import datetime
import hashlib
import json
import platform
import shlex
import sys
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_average(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(x, y):
    x, y = rank_average(x), rank_average(y)
    x, y = x - x.mean(), y - y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def delta(r, k):
    return r * (1 - r) ** 2 * (k - 1) ** 2 / (1 - r + k * r) ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nzs = ["0.005", "0.01", "0.02"]
    imgs = [f"08{i:02d}" for i in range(1, 13)]
    arms = [f"gp_gauss_{s}" for s in (3, 10, 30, 60, 128)] + [
        f"gp_power_{a}" for a in (2.0, 2.5, 3.0)
    ]
    paths = sorted((args.input / "out").glob("d11_n*_i*.json"))
    source = {}
    for path in paths:
        for record in json.loads(path.read_text()):
            key = (str(record["noise"]), record["img"])
            assert key not in source, ("duplicate", key)
            source[key] = record
    assert len(source) == 48
    pred, obs = np.empty((12, 24)), np.empty((12, 24))
    band_excess = np.empty((12, 3))
    labels = [(nz, arm) for nz in nzs for arm in arms]
    for ii, img in enumerate(imgs):
        for jj, (nz, arm) in enumerate(labels):
            row = source[(nz, img)]
            energy = row["E"]
            oracle, current = row["arms"]["gp_oracle"], row["arms"][arm]
            rb = [min(max(1 - oracle["err"][b] / energy[b], 0.0), 1 - 1e-9) for b in range(3)]
            kb = [current["Pb"][b] / oracle["Pb"][b] for b in range(3)]
            pred[ii, jj] = sum(energy[b] * delta(rb[b], kb[b]) for b in range(3))
            obs[ii, jj] = sum(current["err"][b] - oracle["err"][b] for b in range(3))
            if (nz, arm) == ("0.005", "gp_gauss_30"):
                band_excess[ii] = [current["err"][b] - oracle["err"][b] for b in range(3)]
    p, o = pred.mean(0), obs.mean(0)
    rho = spearman(p, o)
    original = json.loads((args.input / "d11_result.json").read_text())
    assert round(rho, 4) == original["D11-b"]["spearman"]
    for jj, point in enumerate(original["D11-b"]["points"]):
        np.testing.assert_allclose(p[jj], point["pred"], rtol=1e-12)
        np.testing.assert_allclose(o[jj], point["obs"], rtol=1e-12)
    # Original RNG seed and original image draws; the first 2000 replicates
    # reproduce the previous run before continuing to protocol-required 10000.
    rng = np.random.default_rng(0)
    draws = rng.choice(len(imgs), size=(10000, len(imgs)))
    bs = np.array([spearman(pred[idx].mean(0), obs[idx].mean(0)) for idx in draws])
    old_ci = np.percentile(bs[:2000], [2.5, 97.5])
    assert [round(float(x), 4) for x in old_ci] == original["D11-b"]["ci"]
    ci = np.percentile(bs, [2.5, 97.5])
    # Additional explicitly separate replicate stream for D11-a; its point
    # statistic is a ratio of pooled image-band excess, not mean image ratios.
    rng_a = np.random.default_rng(0)
    draws_a = rng_a.choice(len(imgs), size=(10000, len(imgs)))
    pooled = band_excess[draws_a].mean(1)
    shares = pooled[:, 2] / pooled.sum(1)
    excess_mean = band_excess.mean(0)
    share = float(excess_mean[2] / excess_mean.sum())
    winner = {}
    for nz in nzs:
        indices = [j for j, label in enumerate(labels) if label[0] == nz]
        pred_best = arms[int(np.argmin(p[indices]))]
        psnrs = [np.mean([source[(nz, img)]["arms"][arm]["psnr"] for img in imgs]) for arm in arms]
        obs_best = arms[int(np.argmax(psnrs))]
        winner[nz] = dict(pred_best=pred_best, obs_best=obs_best, match=pred_best == obs_best)
    arrays_path = args.output / "d11_paired_bootstrap.npz"
    np.savez_compressed(arrays_path, image_ids=np.array(imgs), pred_by_image=pred,
                        obs_by_image=obs, draws=draws, spearman=bs,
                        excess_by_image_band=band_excess, draws_a=draws_a, share_b3=shares)
    result = dict(
        scope="saved-data CPU analysis only; no model inference or new samples",
        protocol="D11-b: same 12 image resamples across all 24 arm/noise points, then average, then Spearman",
        D11_b=dict(spearman=rho, ci_95_percentile=ci.tolist(), bootstrap_replicates=10000,
                   rng="numpy.default_rng(0)", passes_frozen_point_threshold=rho >= 0.8,
                   original_2000_ci_reproduced=old_ci.tolist(),
                   original_ci=original["D11-b"]["ci"],
                   interpretation="point criterion passes; CI lower bound remains below 0.8"),
        D11_a=dict(share_b3=share, ci_95_percentile=np.percentile(shares, [2.5, 97.5]).tolist(),
                   bootstrap_replicates=10000, rng="independent numpy.default_rng(0)",
                   passes_frozen_point_threshold=share >= 0.5,
                   interpretation="point criterion passes; CI crosses 0.5"),
        D11_c=dict(per_noise=winner, n_match=sum(x["match"] for x in winner.values()),
                   passes_frozen_threshold=sum(x["match"] for x in winner.values()) >= 2),
        bootstrap_limitation="conditional on these 12 images, fixed noise draws and measured arms; no noise-draw or training-seed variability estimated",
        diagnostic_limitation="R_b uses truth/oracle error and kappa uses oracle spectrum; this is post-hoc diagnosis, not observable-data-only selection",
        points=[dict(noise=nz, arm=arm, pred=float(p[j]), obs=float(o[j])) for j, (nz, arm) in enumerate(labels)],
    )
    # Scalar identity is second-moment algebra; linear conditional expectation
    # is additional when identifying LMMSE and MMSE (Gaussianity suffices).
    r_grid = np.linspace(0.001, 0.999, 101)[:, None]
    k_grid = np.geomspace(0.01, 100, 121)[None, :]
    gain = k_grid * r_grid / (1 - r_grid + k_grid * r_grid)
    n_over_p = (1 - r_grid) / r_grid
    mse_q_over_p = (1 - gain) ** 2 + gain ** 2 * n_over_p
    excess = mse_q_over_p - (1 - r_grid)
    residual = float(np.max(np.abs(excess - delta(r_grid, k_grid))))
    assert residual < 1e-10
    examples = []
    for rr in ([0.1, 0.9], [0.01, 0.99]):
        exact = float(np.mean(delta(np.array(rr), 4.0)))
        plug = float(delta(np.mean(rr), 4.0))
        examples.append(dict(equal_signal_energies=[1.0, 1.0], R=rr, kappa=[4.0, 4.0],
                             true_energy_normalized_excess=exact, band_average_plugin=plug,
                             plugin_minus_exact=plug - exact,
                             plugin_over_exact=plug / exact))
    theory = dict(
        status="classical LMMSE identity and explicit aggregation counterexamples; not a new estimation theorem",
        scalar_model="y=h*x+n, zero means, E[x*n*]=0, P=E|x|^2>0, noise variance>0, h!=0; z=y/h; N=Var(n)/|h|^2",
        assumed_signal_power="Q=kappa*P, kappa>=0; correct h and correct effective noise N held fixed",
        definitions="R=P/(P+N), a_P=R, a_Q=kappa*R/(1-R+kappa*R)",
        identity="MSE(a_Q)-MSE(a_P)=(P+N)*|a_Q-a_P|^2=P*Delta(R,kappa)",
        delta="R*(1-R)^2*(kappa-1)^2/(1-R+kappa*R)^2",
        gaussian_condition="second moments suffice for LMMSE identity. Identifying it with unrestricted conditional-mean MMSE needs linear conditional expectation; joint Gaussianity is sufficient, not necessary.",
        reciprocal_mismatch="For kappa>1 and 0<R<1, sign(Delta(R,kappa)-Delta(R,1/kappa))=sign(1-2R). Exact symmetry threshold R=0.5 is a scalar-mode statement, not a band-level theorem.",
        reciprocal_difference="R*(1-R)^2*(kappa-1)^2*(kappa^2-1)*(1-2*R)/[(1-R+kappa*R)^2*(kappa*(1-R)+R)^2]",
        limits="For fixed finite kappa>0, Delta->0 as R->0 or R->1. For fixed 0<R<1 and kappa->infinity, Delta->(1-R)^2/R; small R alone does not make arbitrary overestimation harmless.",
        low_R_large_mismatch_example=dict(R=0.01, kappa=100.0, normalized_excess=delta(0.01, 100.0)),
        degeneracies="h=0 is unobserved; zero-noise/zero-prior boundary may be singular. Do not insert singular 0/0 endpoints into the scalar expression.",
        frequency_extension="sum_f P_f*Delta(R_f,kappa_f) is exact only for decoupled scalar modes with correctly specified per-mode observation/noise; general sparse, aliased, rotated/cropped sampling need not be diagonal in the Fourier basis",
        matrix_identity="C_y=A C_x A^H+C_n; K_P=C_x A^H C_y^(-1); excess risk for any fixed linear K_Q is trace((K_Q-K_P) C_y (K_Q-K_P)^H); ^H is Hermitian adjoint",
        band_plugin="sum_b E_b*Delta(mean_R_b,mean_kappa_b) generally differs from sum_f P_f*Delta(R_f,kappa_f). Even energy-weighted means are insufficient.",
        aggregation_examples=examples,
        aggregation_conclusion="examples with identical band mean R=0.5 and kappa=4 give errors of opposite sign, so no universal Jensen upper/lower bound applies; use per-mode values where valid or state the band approximation",
        D11_mapping_limits=[
            "empirical R=1-err_oracle/E is a cropped Hann-windowed finite-sample statistic; it is not guaranteed equal to theoretical P/(P+N)",
            "R clipping preserves the analysis implementation but changes empirical values outside the scalar domain",
            "kappa is a ratio of band PSD sums; the PSD and windowed truth-energy grids differ (512 versus 240)",
            "A includes random rotations, box averages, finite crop and boundary interpolation; Fourier decoupling is not established",
            "output clipping to [0,1], finite CG solves and finite-sample/periodogram effects violate exact fixed-linear scalar interpretation",
        ],
        algebra_check=dict(grid_shape=[101, 121], max_absolute_residual=residual),
    )
    for name, payload in [("d11_bootstrap_10000.json", result), ("scalar_formula_and_band_aggregation.json", theory)]:
        (args.output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    additional = [args.input / name for name in [
        "ana_d11.py", "d11gp.py", "d11_result.json",
        "预注册-D11-E1切换提前诊断-20261005.md", "预注册-D11补充-逐频率R与超额集中-20261005.md"]]
    manifest = dict(
        executed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=platform.python_version(), numpy=np.__version__, host=platform.node(),
        command=shlex.join([sys.executable] + sys.argv),
        script_sha256=sha256(Path(__file__)),
        original_inputs=[dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)) for path in paths + additional],
        outputs=[dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)) for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "execution_manifest.json"],
    )
    (args.output / "execution_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"D11_b": result["D11_b"], "D11_a": result["D11_a"], "D11_c": result["D11_c"],
                      "algebra_check": theory["algebra_check"], "aggregation_examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
