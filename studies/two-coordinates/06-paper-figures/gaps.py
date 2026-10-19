import json, collections, numpy as np
rows = json.load(open('../claude-20261019-two-coordinates-v2/formal/rows.json'))
tot = collections.Counter()
for r in rows: tot[(r['model'], r['seed'], r['s'])] += r['err']['1.0']
ps = lambda e: -10*np.log10(e/32/3)
out = {}
for s in (2, 3, 4, 6):
    v = {m: [ps(tot[(m, sd, s)]) for sd in (1, 2, 3)] for m in ['liif', 'lte', 'sc-inr']}
    out[s] = {m: (float(np.mean(x)), float(np.std(x))) for m, x in v.items()}
    print(f"x{s} LIIF {np.mean(v['liif']):.3f}  LTE-LIIF {np.mean(v['lte'])-np.mean(v['liif']):+.3f}  SC-LIIF {np.mean(v['sc-inr'])-np.mean(v['liif']):+.3f}  seed-std LIIF {np.std(v['liif']):.3f}")
json.dump(out, open('decoder_gaps.json', 'w'), indent=1)
