# F1s analysis: same criteria/cells as claude-20261005-f1/ana_f1.py (prereg frozen), with two fixes:
# (1) Spearman = scipy (ties -> average rank); (2) SC pair = sc-inr vs sc-inr-nosinc (NoSinc config is derived from sc-inr, not sc-inr-eq).
import json, sys, numpy as np
from scipy.stats import spearmanr
RLIN = {2: (1.00, 0.917, 0.085), 3: (0.999, 0.418, 0.010), 4: (0.983, 0.099, 0.012), 8: (0.786, 0.011, 0.008)}
B = ["b1", "b2", "b3"]; DS = ["Set14", "BSD100", "Urban100"]; g = lambda R: R * (1 - R) ** 2
seeds = [int(x) for x in sys.argv[1:]] or [1]
def src(seed, pair, s):
    if seed == 1 and pair == "lte": return "../claude-20261005-f1/out/f1_x%d.json" % s
    return "out/seed%d/f1s_x%d.json" % (seed, s)
PAIRS = {"lte": ("lte", "lte-nocellphase"), "sc": ("sc-inr", "sc-inr-nosinc")}
sp = lambda a, b: float(spearmanr(a, b)[0]); res = {}
for seed in seeds:
    for pn, (full, abl) in PAIRS.items():
        cells = []
        for s in (2, 3, 4, 8):
            d = json.load(open(src(seed, pn, s)))["pooled"]
            for ds in DS:
                for bi, b in enumerate(B):
                    Rf = d["%s|%s|%s" % (ds, full, b)]; cells.append(dict(s=s, ds=ds, b=b, Rlin=RLIN[s][bi], Rnet=Rf, L=Rf - d["%s|%s|%s" % (ds, abl, b)]))
        res[(seed, pn)] = cells
def crit(cells, tag):
    C = [c for c in cells if c["s"] in (2, 3, 4)]
    F1 = sp([g(c["Rlin"]) for c in C], [c["L"] for c in C]); F2 = sp([g(c["Rnet"]) for c in C], [c["L"] for c in C])
    A = np.mean([c["L"] for c in C if c["Rlin"] >= 0.8]); Bz = np.mean([c["L"] for c in C if c["s"] == 3 and c["b"] == "b2"])
    x8 = [c for c in cells if c["s"] == 8]
    print("%-22s H-F1 %.3f  H-F2 %.3f  H-F3 meanL(A)=%.4f L(x3b2)=%.4f -> %s  | x8 g(Rlin) %.3f g(Rnet) %.3f" % (tag, F1, F2, A, Bz, A <= 0.01 and Bz > A, sp([g(c["Rlin"]) for c in x8], [c["L"] for c in x8]), sp([g(c["Rnet"]) for c in x8], [c["L"] for c in x8])))
    return [c["L"] for c in C]
for seed in seeds:
    print("== seed", seed)
    La = crit(res[(seed, "lte")], "lte"); Lb = crit(res[(seed, "sc")], "sc-inr"); print("H-F4 %.3f" % sp(La, Lb))
    for c in res[(seed, "sc")]: print("   sc x%d %-8s %s Rlin=%.3f Rnet=%.3f L=%+.4f" % (c["s"], c["ds"], c["b"], c["Rlin"], c["Rnet"], c["L"]))
if len(seeds) > 1:  # seed-mean cells
    print("== seed mean", seeds)
    m = {}
    for pn in PAIRS:
        cells = [dict(res[(seeds[0], pn)][i], L=float(np.mean([res[(sd, pn)][i]["L"] for sd in seeds])), Rnet=float(np.mean([res[(sd, pn)][i]["Rnet"] for sd in seeds])), Lsd=float(np.std([res[(sd, pn)][i]["L"] for sd in seeds], ddof=1))) for i in range(len(res[(seeds[0], pn)]))]
        m[pn] = crit(cells, pn + " (mean)")
    print("H-F4 %.3f" % sp(m["lte"], m["sc"]))
json.dump({"%d|%s" % k: v for k, v in res.items()}, open("f1s_result_%s.json" % "_".join(map(str, seeds)), "w"), indent=1)
