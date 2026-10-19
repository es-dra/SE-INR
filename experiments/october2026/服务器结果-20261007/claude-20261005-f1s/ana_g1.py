# G1 analysis per paper/derivation/预注册-G1-读出损失与足迹增益偏差-20261005.md (sha256 548b26fe..., frozen 2026-10-04 22:29 UTC)
# groups: {lte, sc} x {seed2, seed3}; SC pair = sc-inr vs sc-inr-nosinc; 27 cells x2/x3/x4; x8 descriptive.
import json, os, numpy as np
from scipy.stats import spearmanr
EPS2 = {"b1": 6e-05, "b2": 0.00552, "b3": 0.07235}
B = ["b1", "b2", "b3"]; DS = ["Set14", "BSD100", "Urban100"]; h = lambda R: R * (1 - 2 * R) ** 2
PAIRS = {"lte": ("lte", "lte-nocellphase"), "sc": ("sc-inr", "sc-inr-nosinc")}
sp = lambda a, b: float(spearmanr(a, b)[0]); G1 = G3 = 0; order = 0; nord = 0; groups = 0; out = {}
for seed in (2, 3):
    for pn, (full, abl) in PAIRS.items():
        fs = ["out/seed%d/f1s_x%d.json" % (seed, s) for s in (2, 3, 4, 8)]
        if not all(os.path.exists(f) for f in fs): print("MISSING group", pn, seed); continue
        cells = []
        for s, f in zip((2, 3, 4, 8), fs):
            d = json.load(open(f))["pooled"]
            for ds in DS:
                for b in B:
                    Rf = d["%s|%s|%s" % (ds, full, b)]; cells.append(dict(s=s, ds=ds, b=b, Rnet=Rf, L=Rf - d["%s|%s|%s" % (ds, abl, b)]))
        C = [c for c in cells if c["s"] in (2, 3, 4)]; L = [c["L"] for c in C]
        r1 = sp([EPS2[c["b"]] for c in C], L); r3 = sp([EPS2[c["b"]] * h(c["Rnet"]) for c in C], L)
        o = sum(1 for s in (2, 3, 4) for ds in DS if [c for c in C if c["s"] == s and c["ds"] == ds and c["b"] == "b3"][0]["L"] > [c for c in C if c["s"] == s and c["ds"] == ds and c["b"] == "b2"][0]["L"])
        x8 = [c for c in cells if c["s"] == 8]; r8 = sp([EPS2[c["b"]] for c in x8], [c["L"] for c in x8])
        groups += 1; G1 += r1 >= 0.5; G3 += (r3 - r1) <= 0.05; order += o; nord += 9
        print("%s seed%d: Spearman(eps2,L)=%.3f  Spearman(eps2*h(Rnet),L)=%.3f diff=%+.3f  L_b3>L_b2 %d/9  | x8 eps2 %.3f" % (pn, seed, r1, r3, r3 - r1, o, r8))
        out["%s|%d" % (pn, seed)] = cells
print("groups present: %d/4" % groups)
print("H-G1: %d groups >= 0.5 -> %s (need >=3)" % (G1, G1 >= 3))
print("H-G2: %d/%d L_b3>L_b2 -> %s (need >=26 of 36)" % (order, nord, order >= 26))
print("H-G3: %d groups diff<=+0.05 -> %s (need >=3)" % (G3, G3 >= 3))
json.dump(out, open("g1_result.json", "w"), indent=1)
