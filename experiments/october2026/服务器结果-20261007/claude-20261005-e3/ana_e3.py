# E3 analysis per 预注册-E3-单图多倍率-冻结-20261005.md (written before any E3 full-run result)
import json, numpy as np
from scipy.stats import spearmanr
D = "/workspace/SE-INR/claude-20261005-e3/"
SC = [2, 3, 8]; BANDS = ["b1", "b2", "b3"]; SETS = ["Set14", "BSD100", "Urban100"]
NETS = ["liif", "lte", "sc-inr-eq", "sc-inr-nosinc"]
prior = {int(r["task"].split("_x")[1].split("_")[0]): r for r in json.load(open(D + "d8_single.json")) if r["task"].endswith("n0.002")}
def zone(R): return "A" if R >= 0.8 else ("C" if R < 0.1 else "B")
res = {s: json.load(open(D + f"out/e3_x{s}.json")) for s in SC}
rng = np.random.default_rng(0)
def boot_img(s, ds, f, B=10000):  # CI of per-image mean of f(rec), resampling images within set
    v = np.array([f(r) for r in res[s]["per_image"] if r["set"] == ds]); idx = rng.integers(0, len(v), (B, len(v))); m = v[idx].mean(1)
    return [round(float(v.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4), len(v)]
cells = []
for s in SC:
    P = res[s]["pooled"]
    for b in BANDS:
        Rp = prior[s][f"R_{b}"]
        for ds in SETS:
            g = lambda k: P[f"{ds}|{k}|{b}"]
            c = dict(scale=s, band=b, set=ds, prior=round(Rp, 4), zone=zone(Rp), generic=g("lmmse_generic"), oracle=g("lmmse_oracle"), bicubic=g("bicubic"), nets={n: g(n) for n in NETS})
            c["net_range"] = max(c["nets"].values()) - min(c["nets"].values())
            c["netmax_minus_oracle"] = max(c["nets"].values()) - c["oracle"]
            c["netmin_minus_oracle"] = min(c["nets"].values()) - c["oracle"]
            c["ci_netmean_minus_oracle(per-image)"] = boot_img(s, ds, lambda r: np.mean([r[f"{n}_{b}"] for n in NETS]) - r[f"lmmse_oracle_{b}"])
            c["ci_oracle_minus_generic(per-image)"] = boot_img(s, ds, lambda r: r[f"lmmse_oracle_{b}"] - r[f"lmmse_generic_{b}"])
            cells.append(c)
assert len(cells) == 27
out = {}
rho = spearmanr([c["prior"] for c in cells], [c["generic"] for c in cells]).correlation
out["H-E3a"] = dict(spearman=round(float(rho), 4), v="成立" if rho >= 0.8 else "不成立")
C = [c for c in cells if c["zone"] == "C"]
okb = [c for c in C if c["netmin_minus_oracle"] > 0 and c["net_range"] <= 0.03]
out["H-E3b"] = dict(n=len(C), n_ok=len(okb), frac=round(len(okb) / len(C), 3), v="成立" if len(okb) / len(C) >= 0.8 else "不成立",
                   fail=[(c["scale"], c["band"], c["set"], round(c["netmin_minus_oracle"], 4), round(c["net_range"], 4)) for c in C if c not in okb])
A = [c for c in cells if c["zone"] == "A"]
okc = [c for c in A if c["netmax_minus_oracle"] <= 0.05]
out["H-E3c"] = dict(n=len(A), n_ok=len(okc), frac=round(len(okc) / len(A), 3), v="成立" if len(okc) / len(A) >= 0.8 else "不成立",
                   fail=[(c["scale"], c["band"], c["set"], round(c["netmax_minus_oracle"], 4)) for c in A if c not in okc])
mono = []
for b in BANDS:
    for ds in SETS:
        r = [next(c["generic"] for c in cells if c["scale"] == s and c["band"] == b and c["set"] == ds) for s in SC]
        mono.append((b, ds, [round(x, 4) for x in r], r[0] > r[1] > r[2]))
out["H-E3d"] = dict(cells=mono, v="成立" if all(m[3] for m in mono) else "不成立")
gap = {z: [c["oracle"] - c["generic"] for c in cells if c["zone"] == z] for z in "ABC"}
mg = {z: round(float(np.mean(v)), 4) if v else None for z, v in gap.items()}
out["H-E3e(探索)"] = dict(mean_gap=mg, n={z: len(v) for z, v in gap.items()}, v="成立" if gap["B"] and mg["B"] > mg["A"] and mg["B"] > mg["C"] else "不成立")
print(json.dumps(out, indent=1, ensure_ascii=False))
for c in cells: print(c["scale"], c["band"], c["set"], c["zone"], c["prior"], "gen", round(c["generic"], 3), "orc", round(c["oracle"], 3), "nets", {k: round(v, 3) for k, v in c["nets"].items()}, "range", round(c["net_range"], 3))
json.dump(dict(cells=cells, verdicts=out), open(D + "e3_result.json", "w"), indent=1, ensure_ascii=False)
