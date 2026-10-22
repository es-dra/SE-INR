import json, collections
A = collections.defaultdict(collections.Counter); T = collections.Counter()
for r in json.load(open('eval_edsr/rows.json')):
    k = (r['s'], r['lo']); A[k].update(r['err']); T[k] += r['tgt']
print({f'x4/{b}': round(1-A[(4, b)]['1.0']/T[(4, b)], 3) for b in (0.25, 0.5, 0.75, 1.0, 1.25)})
