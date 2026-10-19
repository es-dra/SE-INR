# D11 analysis per 预注册-D11-E1切换提前诊断-20261005.md + 预注册-D11补充 (written before any full-run result)
import json, glob, numpy as np
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
D = "/workspace/SE-INR/claude-20261005-d11/out/"
NZ = ["0.0", "0.005", "0.01", "0.02"]; IMG = [f"08{i:02d}" for i in range(1, 13)]
ARMS = [f"gp_gauss_{s}" for s in (3, 10, 30, 60, 128)] + [f"gp_power_{a}" for a in (2.0, 2.5, 3.0)]
J = {}
for nz in NZ:
    for f in glob.glob(D + f"d11_n{nz}_i*.json"):
        for r in json.load(open(f)): J[(nz, r["img"])] = r
for nz in NZ: assert all((nz, i) in J for i in IMG), ("missing", nz)
def Delta(R, k): R = min(max(R, 0.0), 1 - 1e-9); return R * (1 - R) ** 2 * (k - 1) ** 2 / (1 - R + k * R) ** 2
def per_img(nz, i, arm):
    r = J[(nz, i)]; E = r["E"]; eo = r["arms"]["gp_oracle"]["err"]; ea = r["arms"][arm]["err"]
    Rb = [1 - eo[b] / E[b] for b in range(3)]; kb = [r["arms"][arm]["Pb"][b] / r["arms"]["gp_oracle"]["Pb"][b] for b in range(3)]
    pred = sum(E[b] * Delta(Rb[b], kb[b]) for b in range(3)); obs = sum(ea[b] - eo[b] for b in range(3))
    return pred, obs, [ea[b] - eo[b] for b in range(3)], Rb, kb
rng = np.random.default_rng(0); out = {}
NB = ["0.005", "0.01", "0.02"]
def points(imgs):
    P, O, lab = [], [], []
    for nz in NB:
        for arm in ARMS:
            v = [per_img(nz, i, arm) for i in imgs]; P.append(np.mean([x[0] for x in v])); O.append(np.mean([x[1] for x in v])); lab.append((nz, arm))
    return P, O, lab
P, O, lab = points(IMG); rho = spearmanr(P, O).correlation
bs = []
for _ in range(2000):
    s = list(rng.choice(IMG, len(IMG))); p, o, _ = points(s); bs.append(spearmanr(p, o).correlation)
out["D11-b"] = dict(spearman=round(float(rho), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], v="成立" if rho >= 0.8 else "不成立",
                    points=[dict(noise=l[0], arm=l[1], pred=float(p), obs=float(o)) for l, p, o in zip(lab, P, O)])
# D11-a
def share_b3(imgs):
    ex = np.mean([per_img("0.005", i, "gp_gauss_30")[2] for i in imgs], 0); return float(ex[2] / ex.sum()), ex.tolist()
s3, ex = share_b3(IMG); bs = [share_b3(list(rng.choice(IMG, len(IMG))))[0] for _ in range(10000)]
Rb = np.mean([per_img("0.005", i, "gp_gauss_30")[3] for i in IMG], 0).tolist(); kb = np.mean([per_img("0.005", i, "gp_gauss_30")[4] for i in IMG], 0).tolist()
out["D11-a"] = dict(share_b3=round(s3, 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], excess_by_band=ex, R_b_oracle=Rb, kappa_b_gauss30=kb, v="成立" if s3 >= 0.5 else "不成立")
# D11-c
c = {}
for nz in NB:
    pm = {a: np.mean([per_img(nz, i, a)[0] for i in IMG]) for a in ARMS}; ps = {a: np.mean([J[(nz, i)]["arms"][a]["psnr"] for i in IMG]) for a in ARMS}
    c[nz] = dict(pred_best=min(pm, key=pm.get), obs_best=max(ps, key=ps.get)); c[nz]["match"] = c[nz]["pred_best"] == c[nz]["obs_best"]
nm = sum(v["match"] for v in c.values()); out["D11-c"] = dict(per_noise=c, n_match=nm, v="成立" if nm >= 2 else "不成立")
# D11-d / D11-e (per-frequency)
f = np.fft.fftfreq(240); m = np.maximum(np.abs(f)[:, None], np.abs(f)[None]); BW = [m < 1 / 8, (m >= 1 / 8) & (m < 1 / 4), m >= 1 / 4]
def spec(nz, i): return np.load(D + f"d11_n{nz}_i{int(i) - 801}_{i}_spec.npz")
dd = {b: dict(frac_pts=[], frac_energy=[]) for b in range(3)}; conc = {a: dict(all=[], b1=[], b2=[], b3=[]) for a in ("gp_gauss_30", "gp_power_2.0", "gp_power_3.0")}; base = []
for i in IMG:
    s5, s0 = spec("0.005", i), spec("0.0", i)
    Ps = gaussian_filter(s5["truth"].astype(np.float64), 2, mode="wrap"); eo = gaussian_filter(s5["gp_oracle"].astype(np.float64), 2, mode="wrap")
    low = (1 - eo / Ps) < 0.5; base.append(float(low.mean()))
    for b in range(3):
        dd[b]["frac_pts"].append(float(low[BW[b]].mean())); dd[b]["frac_energy"].append(float(Ps[BW[b] & low].sum() / Ps[BW[b]].sum()))
    for a in conc:
        de = np.maximum(s5[a].astype(np.float64) - s0[a].astype(np.float64), 0)
        conc[a]["all"].append(float(de[low].sum() / de.sum()))
        for b in range(3): conc[a][f"b{b+1}"].append(float(de[BW[b] & low].sum() / max(de[BW[b]].sum(), 1e-30)))
def mci(v):
    v = np.array(v); idx = rng.integers(0, len(v), (10000, len(v))); mm = v[idx].mean(1); return [round(float(v.mean()), 4), round(float(np.percentile(mm, 2.5)), 4), round(float(np.percentile(mm, 97.5)), 4)]
out["D11-d"] = {f"b{b+1}": dict(frac_pts=mci(dd[b]["frac_pts"]), frac_energy=mci(dd[b]["frac_energy"])) for b in range(3)}
out["D11-d"]["all_frac_pts_baseline"] = mci(base)
C = {a: {k: mci(v) for k, v in d.items()} for a, d in conc.items()}
ok = C["gp_gauss_30"]["all"][0] > 0.5 and C["gp_power_2.0"]["all"][0] > 0.5 and C["gp_power_3.0"]["all"][0] < min(C["gp_gauss_30"]["all"][0], C["gp_power_2.0"]["all"][0])
out["D11-e"] = dict(conc=C, v="成立" if ok else "不成立")
print(json.dumps({k: (v if k != "D11-b" else {kk: vv for kk, vv in v.items() if kk != "points"}) for k, v in out.items()}, indent=1, ensure_ascii=False))
for p in out["D11-b"]["points"]: print(p["noise"], p["arm"], "pred %.1f obs %.1f" % (p["pred"], p["obs"]))
json.dump(out, open(D + "../d11_result.json", "w"), indent=1, ensure_ascii=False)
