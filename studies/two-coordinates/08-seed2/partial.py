import json, collections, numpy as np
def ps(p):
    A = collections.Counter()
    for r in json.load(open(p)): A[r['s']] += r['err']['1.0']
    return {s: -10*np.log10(v/32/3) for s, v in A.items()}
e = ps('eval_edsr/rows.json')
for a in ['tinydec', 'nocell']:
    x = ps(f'eval_{a}/rows.json'); print(a, {s: round(x[s]-e[s], 4) for s in (2, 3, 4, 6)})
print('edsr seed2 psnr', {s: round(v, 3) for s, v in e.items()})
