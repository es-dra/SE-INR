# E1 analysis per 分析口径-跑前写定.md (written before any E1 result)
# v2 (2026-10-04): fix glob bug in v1 (n0.0* also matched n0.005/n0.01/n0.02 files); rules unchanged
import json, glob, numpy as np
D = "/workspace/SE-INR/claude-20261004-e1/out/"
LOW = ["mrgrid", "ls_tv1e-4", "gp_gauss_30", "gp_gauss_60", "gp_gauss_128"]
PRI = ["ff_3", "ff_10", "ls_tv1e-3", "gp_gauss_3", "gp_gauss_10", "gp_power_2.0", "gp_power_2.5", "gp_power_3.0", "gp_oracle"]
NZ = ["0.0", "0.005", "0.01", "0.02"]
OK, UNC, NO = "成立", "不确定", "不成立"


def table(T):
    tab = {}
    for nz in NZ:
        rows = {}
        for f in sorted(glob.glob(D + f"ff_T{T}_n{nz}.json")) + sorted(glob.glob(D + f"ff_T{T}_n{nz}_s*.json")) + sorted(glob.glob(D + f"gp_T{T}_n{nz}.json")):
            for r in json.load(open(f)):
                rows.setdefault(r["img"], {}).update({k: v for k, v in r.items() if isinstance(v, float) and k != "noise"})
        tab[nz] = rows
    return tab


rng = np.random.default_rng(0)


def boot(d, B=10000):
    d = np.asarray(d); idx = rng.integers(0, len(d), (B, len(d))); m = d[idx].mean(1)
    return [round(float(d.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def judge_ge(ci, thr):
    m, lo, hi = ci
    return OK if lo >= thr else (UNC if m >= thr else NO)


def vec(rows, k, imgs): return np.array([rows[i][k] for i in imgs])
def best(rows, ks, imgs): return max(ks, key=lambda k: vec(rows, k, imgs).mean())


means = {}
for T in (16, 8, 4):
    tab = table(T); means[T] = {}
    for nz in NZ:
        rows = tab[nz]; imgs = sorted(i for i in rows if all(k in rows[i] for k in LOW + PRI))
        if not imgs: continue
        means[T][nz] = dict(n=len(imgs), imgs=imgs, mean={k: round(float(vec(rows, k, imgs).mean()), 3) for k in LOW + PRI + ["bilinear"]})
print(json.dumps(means, indent=1, ensure_ascii=False))

tab = table(16); res = {}
def g(nz):
    rows = tab[nz]; imgs = sorted(i for i in rows if all(k in rows[i] for k in LOW + PRI))
    assert len(imgs) == 12, (nz, imgs)
    return rows, imgs

p1 = {}
for nz in ("0.0", "0.005"):
    rows, imgs = g(nz); lb = best(rows, ["mrgrid", "gp_gauss_30", "gp_gauss_60", "gp_gauss_128"], imgs); ff = best(rows, ["ff_3", "ff_10"], imgs)
    ci = boot(vec(rows, lb, imgs) - vec(rows, ff, imgs)); p1[nz] = dict(low=lb, ff=ff, diff=ci, v=judge_ge(ci, 1.0))
p1["v"] = OK if all(p1[n]["v"] == OK for n in ("0.0", "0.005")) else NO
res["P1"] = p1

rows, imgs = g("0.02"); ff = best(rows, ["ff_3", "ff_10"], imgs); pw = best(rows, ["gp_power_2.0", "gp_power_2.5", "gp_power_3.0"], imgs)
ci = boot(vec(rows, ff, imgs) - vec(rows, "mrgrid", imgs)); a = dict(ff=ff, diff=ci, v=judge_ge(ci, 3.0))
b = {}
for x, y in ((ff, pw), (ff, "gp_oracle"), (pw, "gp_oracle")):
    ci = boot(vec(rows, x, imgs) - vec(rows, y, imgs))
    b[f"{x} - {y}"] = dict(diff=ci, v=OK if (ci[1] >= -0.6 and ci[2] <= 0.6) else (UNC if abs(ci[0]) <= 0.6 else NO))
res["P2"] = dict(a=a, b=b, v=OK if a["v"] == OK and all(x["v"] == OK for x in b.values()) else NO)

Dv = {}
for nz in ("0.005", "0.02"):
    rows, imgs = g(nz); pw = best(rows, ["gp_power_2.0", "gp_power_2.5", "gp_power_3.0"], imgs)
    Dv[nz] = (pw, vec(rows, "gp_oracle", imgs) - vec(rows, pw, imgs), imgs)
assert Dv["0.005"][2] == Dv["0.02"][2]
ci = boot(Dv["0.02"][1] - Dv["0.005"][1])
res["P3"] = dict(power_best={k: v[0] for k, v in Dv.items()}, D={k: round(float(v[1].mean()), 3) for k, v in Dv.items()}, diff=ci, v=judge_ge(ci, 0.0))

win = {}
for nz in NZ:
    rows, imgs = g(nz); ks = sorted(LOW + PRI, key=lambda k: -vec(rows, k, imgs).mean()); w, s = ks[0], ks[1]
    win[nz] = dict(winner=w, group="LOW" if w in LOW else "PRI", second=s, gap=boot(vec(rows, w, imgs) - vec(rows, s, imgs)))
grp = [win[nz]["group"] for nz in NZ]; first = next((NZ[i] for i, x in enumerate(grp) if x == "PRI"), None)
res["P4"] = dict(win=win, first_switch=first, v=OK if (grp[0] == "LOW" and grp[-1] == "PRI" and first in ("0.005", "0.01", "0.02")) else NO)
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(dict(means=means, verdicts=res), open(D + "../e1_result_v2.json", "w"), indent=1, ensure_ascii=False)
