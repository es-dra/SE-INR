"""Retrospective paired-image controls from original float32 sufficient statistics.

No models are loaded or called. h is the change in EACH rel_cell coordinate,
not the Euclidean norm of the two-dimensional change. Ratios are ratios of
pooled quantities, never means of per-model or per-image ratios.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = {"Set14": 14, "BSD100": 100, "Urban100": 100}
MODELS = ["liif", "lte", "sc-inr"]
SEEDS = [1, 2]
SCALES = [2, 3, 4, 6]
MUS = [0.5, 0.625, 0.75, 0.8, 0.9091, 1.1, 1.25, 1.3333, 1.6, 2.0]
RAW_KEYS = ["J", "A", "D", "V"]
EXPECTED_SOURCE_SHA = "f723dac9241f89c6eb344101cd137d2356136621cc2c8b3c70f31f2071b30999"
METRICS = ["J", "A", "cross", "D", "V", "cost_db", "J_per_h2", "D_per_h2", "cross_per_h2"]
CONTRASTS = [
    ("matched_negative_h_-0.25", (2, 0.75), (3, 0.625)),
    ("matched_negative_h_-0.25", (3, 0.625), (4, 0.5)),
    ("matched_negative_h_-0.25", (2, 0.75), (4, 0.5)),
    ("matched_positive_h_approx_1over3", (2, 1.3333), (6, 2.0)),
]
CONTRASTS += [("same_multiplier", (s, mu), (t, mu))
              for mu in MUS for s, t in [(2, 3), (3, 4), (4, 6), (2, 6)]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def h(scale, mu):
    return 2 * (mu - 1) / scale


def condition(scale, mu, prefix=""):
    result = {
        "scale": scale, "mu": mu, "h": h(scale, mu),
        "effective_scale_s_over_mu": scale / mu,
        "native_in_training_scale_range": 1 <= scale <= 4,
        "perturbed_cell_in_training_scale_range": 1 <= scale / mu <= 4,
    }
    return {prefix + k: v for k, v in result.items()}


def load(path):
    obj = json.loads(gzip.decompress(path.read_bytes()))
    assert obj["source_sha256"] == EXPECTED_SOURCE_SHA
    assert obj["model_calls"] == 0
    assert len(obj["records"]) == 214 * 3 * 2 * 4 * 10
    assert obj["rows"] == 214 * 3 * 2 * sum(4*s for s in SCALES)
    assert obj["checked_cases"] == obj["rows"] * 10
    arrays, names, checks = {}, {}, {}
    for ds, expected in DATASETS.items():
        records = [r for r in obj["records"] if r["dataset"] == ds]
        images = sorted({r["image"] for r in records})
        assert len(images) == expected
        assert len(records) == expected * 3 * 2 * 4 * 10
        image_index = {name: i for i, name in enumerate(images)}
        x = np.full((expected, 3, 2, 4, 10, 4), np.nan)
        energy = np.full((expected, 3, 2, 4, 10), np.nan)
        for r in records:
            assert r["scope"] == "all_nonDC"
            assert r["bins"] == 4 * r["scale"]
            idx = (image_index[r["image"]], MODELS.index(r["model"]),
                   SEEDS.index(r["seed"]), SCALES.index(r["scale"]), MUS.index(r["mu"]))
            assert np.isnan(x[idx]).all(), ("duplicate", ds, idx)
            # Upstream cross stores A=<T-P, P_m-P>, not its MSE contribution.
            x[idx] = [r["J"], r["cross"], r["D"], r["V"]]
            energy[idx] = r["E"]
        assert np.isfinite(x).all() and np.isfinite(energy).all()
        J, A, D, V = np.moveaxis(x, -1, 0)
        identity = float(np.max(np.abs(D-J+2*A)))
        np.testing.assert_allclose(D, J-2*A, rtol=1e-9, atol=1e-12)
        assert np.min(J) >= 0 and np.min(V) > 0 and np.min(V+D) > 0
        np.testing.assert_allclose(V, np.broadcast_to(V[..., :1], V.shape), atol=1e-14, rtol=1e-11)
        e0 = energy[:, :1, :1, :1, :1]
        np.testing.assert_allclose(energy, np.broadcast_to(e0, energy.shape), atol=1e-14, rtol=1e-11)
        arrays[ds], names[ds] = x, images
        checks[ds] = dict(images=expected, records=len(records), complete_grid=True,
                          identity_max_abs=identity, V_constant_over_mu=True,
                          target_energy_constant_across_conditions=True)
    extract_path = ROOT / "extract_all_response.py"
    extractor_matches = sha(extract_path) == obj["script_sha256"]
    assert extractor_matches
    provenance = {k: v for k, v in obj.items() if k != "records"}
    provenance.update(input_path=str(path.resolve()), input_sha256=sha(path),
                      analysis_script_sha256=sha(Path(__file__)),
                      extractor_matches_recorded_sha256=extractor_matches,
                      original_source_sha_checked_against_expected=True,
                      original_source_bytes_rehashed_here=False)
    return arrays, names, checks, provenance


def groups(x):
    """Last axes are model, seed, scale, mu, stat; preceding axes are batches."""
    result = [("mean_of_models", "mean_of_seeds", x.mean(axis=-4).mean(axis=-4))]
    for m, model in enumerate(MODELS):
        result.append((model, "mean_of_seeds", x[..., m, :, :, :, :].mean(axis=-4)))
        for k, seed in enumerate(SEEDS):
            result.append((model, seed, x[..., m, k, :, :, :]))
    return result


def measure(x):
    J, A, D, V = np.moveaxis(x, -1, 0)
    hs = np.array([[h(s, mu) for mu in MUS] for s in SCALES])
    return dict(J=J, A=A, cross=-2*A, D=D, V=V,
                cost_db=10*np.log10((V+D)/V), J_per_h2=J/hs**2,
                D_per_h2=D/hs**2, cross_per_h2=-2*A/hs**2)


def ci(x):
    lo, hi = np.quantile(x, [0.025, 0.975], axis=0)
    return float(lo), float(hi)


def interval_fields(value, draws, prefix):
    low, high = ci(draws)
    return {prefix: float(value), prefix + "_ci_low": low, prefix + "_ci_high": high}


def bootstrap(x, count, seed):
    # The same image multiplicities apply to all models, seeds, scales and cells.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), (count, len(x)))
    weights = np.stack([np.bincount(row, minlength=len(x)) for row in indices]) / len(x)
    draws = np.tensordot(weights, x, axes=(1, 0))
    return draws, hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()


def summaries(ds, point, samples):
    scale_rows, pair_rows, trend_rows = [], [], []
    for (model, seed, x), (bm, bs, xb) in zip(groups(point), groups(samples)):
        assert (model, seed) == (bm, bs)
        v, b = measure(x), measure(xb)
        label = dict(dataset=ds, model=model, seed=seed)
        for si, s in enumerate(SCALES):
            for mi, mu in enumerate(MUS):
                row = dict(**label, **condition(s, mu))
                for key in METRICS:
                    row.update(interval_fields(v[key][si, mi], b[key][:, si, mi], key))
                scale_rows.append(row)
        for family, start, end in CONTRASTS:
            i = (SCALES.index(start[0]), MUS.index(start[1]))
            j = (SCALES.index(end[0]), MUS.index(end[1]))
            row = dict(**label, family=family,
                       **condition(*start, prefix="start_"), **condition(*end, prefix="end_"))
            row["relative_h_difference_end_over_start_minus1"] = h(*end)/h(*start)-1
            for key in METRICS:
                pv0, pv1 = v[key][i], v[key][j]
                bv0, bv1 = b[key][(slice(None),)+i], b[key][(slice(None),)+j]
                row.update(interval_fields(pv0, bv0, key+"_start"))
                row.update(interval_fields(pv1, bv1, key+"_end"))
                row.update(interval_fields(pv1-pv0, bv1-bv0, key+"_end_minus_start"))
                # dB is already logarithmic: use its paired difference, no dB ratio.
                if key != "cost_db":
                    assert np.all(bv0 != 0) and pv0 != 0
                    row.update(interval_fields(pv1/pv0, bv1/bv0, key+"_ratio_end_over_start"))
                    low, high = ci(bv0)
                    row[key+"_ratio_denominator_ci_contains_zero"] = low <= 0 <= high
            np.testing.assert_allclose(row["D_end_minus_start"],
                                       row["J_end_minus_start"]+row["cross_end_minus_start"],
                                       atol=1e-12, rtol=1e-9)
            pair_rows.append(row)
        sequences = [("matched_negative_h_-0.25", [(2,.75),(3,.625),(4,.5)]),
                     ("matched_positive_h_approx_1over3", [(2,1.3333),(6,2.)])]
        sequences += [("same_multiplier_"+str(mu), [(s,mu) for s in SCALES]) for mu in MUS]
        for family, sequence in sequences:
            row = dict(**label, family=family, conditions=json.dumps(sequence))
            for key in METRICS:
                values = np.array([v[key][SCALES.index(s),MUS.index(mu)] for s,mu in sequence])
                diffs = np.diff(values)
                row[key+"_nonincreasing"] = bool(np.all(diffs <= 0))
                row[key+"_nondecreasing"] = bool(np.all(diffs >= 0))
                row[key+"_trend"] = ("decreasing" if np.all(diffs < 0) else
                                       "increasing" if np.all(diffs > 0) else "nonmonotonic")
            trend_rows.append(row)
    return scale_rows, pair_rows, trend_rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(path, scale_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    colors = {"Set14": "#137a91", "BSD100": "#d2693c", "Urban100": "#7b5a9e"}
    sequences = [[(s,2.) for s in SCALES], [(2,.75),(3,.625),(4,.5)], [(2,1.3333),(6,2.)]]
    titles = ["Fixed multiplier m = 2", "Matched h = −0.25", "Matched h ≈ 1/3 (0.01% mismatch)"]
    for col, (sequence, title) in enumerate(zip(sequences, titles)):
        for ds in DATASETS:
            rows = [next(r for r in scale_rows if r["dataset"] == ds and r["model"] == "mean_of_models"
                         and r["scale"] == s and r["mu"] == mu) for s,mu in sequence]
            sx = [s for s,mu in sequence]
            for ax, key in [(axes[0,col],"J_per_h2"), (axes[1,col],"D_per_h2")]:
                ax.plot(sx,[r[key] for r in rows],marker="o",color=colors[ds],label=ds)
                ax.fill_between(sx,[r[key+"_ci_low"] for r in rows],
                                [r[key+"_ci_high"] for r in rows],color=colors[ds],alpha=.15)
                ax.set_xticks(sx); ax.grid(alpha=.15); ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        axes[0,col].set_title(title,fontsize=10)
        axes[1,col].set_xlabel("Upsampling scale s")
        axes[1,col].axhline(0,color="gray",lw=.7)
    axes[0,0].set_ylabel("Output-change energy J / h²")
    axes[1,0].set_ylabel("Net MSE cost D / h²")
    axes[0,0].legend(frameon=False)
    fig.suptitle("Cell step controls from saved float32 statistics",fontsize=14)
    fig.text(.5,.015,"Bands: all non-DC. Equal image / fixed-seed / model weights within each dataset. Shading: paired image bootstrap 95% CI.\n"
             "h is per-axis rel_cell change. Cross-scale controls also change LR features, native cell and training support.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95))
    fig.savefig(path/"step-controls.png",dpi=180)
    fig.savefig(path/"step-controls.pdf")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=ROOT/"inputs/response-all/response_stats.json.gz")
    p.add_argument("--out",type=Path,default=ROOT/"step-controls-v1")
    p.add_argument("--bootstrap",type=int,default=2000)
    a=p.parse_args()
    assert a.bootstrap >= 2000
    assert not a.out.exists(), "Choose a new output directory; frozen outputs are not overwritten."
    arrays,names,checks,provenance=load(a.input)
    rows,pairs,trends,bootinfo=[],[],[],{}
    for dsi,(ds,x) in enumerate(arrays.items()):
        draws,digest=bootstrap(x,a.bootstrap,2026102301+dsi)
        sr,pr,tr=summaries(ds,x.mean(axis=0),draws)
        rows+=sr; pairs+=pr; trends+=tr
        bootinfo[ds]=dict(samples=a.bootstrap,rng_seed=2026102301+dsi,
                          image_index_sha256=digest,images=names[ds])
    primary=[r for r in pairs if r["model"]=="mean_of_models" and
             (r["family"].startswith("matched_") or
              (r["start_scale"]==2 and r["end_scale"]==6 and r["start_mu"]==2.))]
    report=dict(
        provenance=provenance,checks=checks,bootstrap=bootinfo,
        definitions={
            "J":"mean ||P_m-P||^2, original RGB all-non-DC normalization",
            "A":"mean Re<T-P,P_m-P>; source field named cross",
            "cross":"-2A, signed contribution to net MSE cost",
            "D":"MSE(P_m,T)-MSE(P,T)=J+cross",
            "V":"native MSE over the same all-non-DC frequencies",
            "cost_db":"10 log10((mean V + mean D)/mean V), positive is harm; not mean per-image PSNR difference",
            "h":"2(mu-1)/s, change in each of the two rel_cell coordinates; not vector norm",
            "aggregation":"Within each dataset, equal image and fixed-seed means, then equal model mean. Ratios use these pooled values.",
            "ci":"Percentile paired image bootstrap; each resampled image carries all models, seeds, scales and multipliers. Conditions on the two existing seeds.",
            "ratio_warning":"Ratios with denominator CI spanning zero are descriptive and unstable; use paired differences.",
        },
        interpretation_limits=[
            "Retrospective saved-data comparison; no new training or model calls.",
            "Matched per-axis h does not match native cell, LR features, local coordinates, or train support across scale; not a causal Jacobian isolation.",
            "For positive matched steps, h at (2,1.3333) is 0.3333 and h at (6,2) is 1/3: end/start-1 = 0.0100010001 percent.",
            "Positive matched start has native and perturbed scales in [1,4]; x6 native is out of range and perturbed effective scale 3 is in range.",
            "Negative matched native scales 2,3,4 are in range; perturbed effective scales are 8/3,4.8,8, so latter two are out of range.",
            "All quantities use original all-non-DC cropped frequency rows; not full-image or DC-inclusive benchmark MSE.",
        ],
        primary_contrasts=primary,scales=rows,pairs=pairs,trends=trends,model_calls=0)
    a.out.mkdir()
    (a.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    write_csv(a.out/"scales.csv",rows)
    write_csv(a.out/"pairs.csv",pairs)
    write_csv(a.out/"trends.csv",trends)
    write_csv(a.out/"primary_contrasts.csv",primary)
    write_csv(a.out/"model_seed_scales.csv",[r for r in rows if isinstance(r["seed"],int)])
    write_csv(a.out/"model_seed_pairs.csv",[r for r in pairs if isinstance(r["seed"],int)])
    plot(a.out,rows)
    print(json.dumps(dict(output=str(a.out),checks=checks,bootstrap=a.bootstrap,model_calls=0)))


if __name__=="__main__":
    main()
