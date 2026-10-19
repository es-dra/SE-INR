# F1 analysis, criteria fixed in 预注册-F1-读出算子价值与三区-20261005.md
import json, numpy as np
RLIN = {2: (1.00, 0.917, 0.085), 3: (0.999, 0.418, 0.010), 4: (0.983, 0.099, 0.012), 8: (0.786, 0.011, 0.008)}
B = ["b1", "b2", "b3"]; DS = ["Set14", "BSD100", "Urban100"]
def src(pair, s):
    if pair == "lte": return "out/f1_x%d.json" % s
    return "../claude-20260926/h6c.json" if s == 4 else "../claude-20261005-e3/out/e3_x%d.json" % s
PAIRS = {"lte": ("lte", "lte-nocellphase"), "sc": ("sc-inr-eq", "sc-inr-nosinc")}
g = lambda R: R * (1 - R) ** 2
def rank(x): return np.argsort(np.argsort(x)).astype(float)
def spear(a, b): return float(np.corrcoef(rank(np.array(a)), rank(np.array(b)))[0, 1])
rng = np.random.default_rng(20261005); out = {}
for pn, (full, abl) in PAIRS.items():
    cells = []
    for s in (2, 3, 4, 8):
        d = json.load(open(src(pn, s)))
        for ds in DS:
            imgs = [r for r in d["per_image"] if r["set"] == ds]
            for bi, b in enumerate(B):
                Rf = d["pooled"]["%s|%s|%s" % (ds, full, b)]; Ra = d["pooled"]["%s|%s|%s" % (ds, abl, b)]
                li = np.array([r["%s_%s" % (full, b)] - r["%s_%s" % (abl, b)] for r in imgs])
                bs = [li[rng.integers(0, len(li), len(li))].mean() for _ in range(10000)]
                cells.append(dict(s=s, ds=ds, b=b, Rlin=RLIN[s][bi], Rnet=Rf, L=Rf - Ra, ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))))
    out[pn] = cells
    C = [c for c in cells if c["s"] in (2, 3, 4)]
    print("== pair", pn, full, "vs", abl)
    for c in cells: print("x%d %-8s %s Rlin=%.3f Rnet=%.3f L=%+.4f  per-image CI [%+.4f, %+.4f]" % (c["s"], c["ds"], c["b"], c["Rlin"], c["Rnet"], c["L"], *c["ci"]))
    F1 = spear([g(c["Rlin"]) for c in C], [c["L"] for c in C]); F2 = spear([g(c["Rnet"]) for c in C], [c["L"] for c in C])
    A = [c["L"] for c in C if c["Rlin"] >= 0.8]; Bz = [c["L"] for c in C if c["s"] == 3 and c["b"] == "b2"]
    print("H-F1 Spearman(g(Rlin),L)=%.3f -> %s" % (F1, F1 >= 0.5)); print("H-F2 Spearman(g(Rnet),L)=%.3f -> %s" % (F2, F2 >= 0.5))
    print("H-F3 meanL(A)=%.4f (<=0.01: %s); L(x3 b2)=%.4f > meanL(A): %s" % (np.mean(A), np.mean(A) <= 0.01, np.mean(Bz), np.mean(Bz) > np.mean(A)))
    x8 = [c for c in cells if c["s"] == 8]; print("x8 (OOD, descriptive) Spearman g(Rlin)=%.3f g(Rnet)=%.3f" % (spear([g(c["Rlin"]) for c in x8], [c["L"] for c in x8]), spear([g(c["Rnet"]) for c in x8], [c["L"] for c in x8])))
La = [c["L"] for c in out["lte"] if c["s"] in (2, 3, 4)]; Lb = [c["L"] for c in out["sc"] if c["s"] in (2, 3, 4)]
print("H-F4 Spearman(L_lte, L_sc)=%.3f -> %s" % (spear(La, Lb), spear(La, Lb) >= 0.5))
json.dump(out, open("f1_result.json", "w"), indent=1)
