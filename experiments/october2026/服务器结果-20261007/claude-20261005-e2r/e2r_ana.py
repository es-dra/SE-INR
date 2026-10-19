# E2 reproduction analysis: criteria H-E2a..d exactly as in 预注册-E2-E1 §2
import json, glob, numpy as np
from scipy.stats import spearmanr
D = "/workspace/SE-INR/claude-20261005-e2r/out/"
TASKS = ["single2", "single4", "sparse10", "sparse25", "sparse50"]; K = [1 / 16, 1 / 4, 4, 16]
PRIOR = {"single2": [1.00, .90, .08], "single4": [.98, .08, .01], "sparse10": [.81, .20, .07], "sparse25": [.92, .47, .20], "sparse50": [.98, .76, .46]}
def Delta(R, k): R = min(max(R, 0.0), 1 - 1e-9); return R * (1 - R) ** 2 * (k - 1) ** 2 / (1 - R + k * R) ** 2
rs = [json.load(open(f)) for f in sorted(glob.glob(D + "*.json"))]
cells = []; combos = {}
for t in TASKS:
    rt = [r for r in rs if r["task"] == t]; assert len(rt) == 12, (t, len(rt))
    for b in range(3):
        mR = float(np.mean([r["matched"][b] for r in rt])); cost = {}
        for k in K:
            obs = float(np.mean([r["matched"][b] - r["mis"][f"b{b+1}_{k}"][b] for r in rt]))
            post = float(np.mean([Delta(r["matched"][b], k) for r in rt])); pri = Delta(PRIOR[t][b], k)
            cells.append(dict(task=t, band=f"b{b+1}", kappa=k, R=mR, obs=obs, post=post, prior=pri)); cost[k] = obs
        combos[(t, b)] = (mR, cost, float(np.mean([r["generic"][b] for r in rt])))
assert len(cells) == 60
o = [c["obs"] for c in cells]; out = {}
sa = spearmanr([c["post"] for c in cells], o).correlation; out["H-E2a"] = dict(spearman=round(float(sa), 3), pearson=round(float(np.corrcoef([c["post"] for c in cells], o)[0, 1]), 3), v="成立" if sa >= .8 else "不成立")
lo = [(t, b) for (t, b), v in combos.items() if v[0] < .3]; hi = [(t, b) for (t, b), v in combos.items() if v[0] > .7]
nlo = sum(combos[c][1][16] > combos[c][1][1 / 16] for c in lo); nhi = sum(combos[c][1][1 / 16] > combos[c][1][16] for c in hi)
out["H-E2b"] = dict(low=f"{nlo}/{len(lo)}", high=f"{nhi}/{len(hi)}", v="成立" if (nlo / max(len(lo), 1) >= .8 and nhi / max(len(hi), 1) >= .8) else "不成立",
                    low_fail=[c for c in lo if not combos[c][1][16] > combos[c][1][1 / 16]], high_fail=[c for c in hi if not combos[c][1][1 / 16] > combos[c][1][16]])
err = np.abs(np.array([c["post"] for c in cells]) - np.array(o)); out["H-E2c"] = dict(median=round(float(np.median(err)), 4), max=round(float(err.max()), 3), v="成立" if np.median(err) <= .05 else "不成立")
sd = spearmanr([c["prior"] for c in cells], o).correlation; out["H-E2d"] = dict(spearman=round(float(sd), 3), v="成立" if sd >= .7 else "不成立")
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
for (t, b), (mR, cost, g) in combos.items(): print(t, f"b{b+1}", "R=%.3f generic=%.3f" % (mR, g), {str(k): round(v, 3) for k, v in cost.items()})
json.dump(dict(cells=cells, verdicts=out), open(D + "../e2r_result.json", "w"), indent=1, ensure_ascii=False, default=str)
