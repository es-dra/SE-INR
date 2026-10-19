"""Descriptive: per-u-bin cost of cell m vs native, pooled over datasets/seeds/images, per model x scale. Also native R_b and error share."""
import json, sys, collections, numpy as np
rows = json.load(open(sys.argv[1]))
A = collections.defaultdict(lambda: collections.Counter()); T = collections.Counter()
for r in rows:
    k = (r['model'], r['s'], r['lo'])
    for m, v in r['err'].items(): A[k][m] += v
    T[k] += r['tgt']
for model in ['liif', 'lte', 'sc-inr']:
    for s in [2, 3, 4, 6]:
        ks = sorted(k for k in A if k[0] == model and k[1] == s)
        if not ks: continue
        nat = sum(A[k]['1.0'] for k in ks)
        print(f'\n== {model} x{s}  cost in % of total native MSE per u-bin;  R=native recovery')
        print('u-bin   share   R     ' + ' '.join(f'{m:>7}' for m in ['0.5', '0.75', '0.9091', '1.1', '1.3333', '2.0']))
        merged = collections.OrderedDict()
        for k in ks:
            lo = k[2]; key = lo if lo < 2 else (2.0 if lo < 3 else 3.0 if lo < 4 else 4.0)
            mc = merged.setdefault(key, [collections.Counter(), 0.0])
            mc[0].update(A[k]); mc[1] += T[k]
        for key, (c, t) in merged.items():
            print(f'{key:4.2f}  {100*c["1.0"]/nat:5.1f}% {1-c["1.0"]/t:5.2f} ' + ' '.join(f'{100*(c[m]-c["1.0"])/nat:+7.2f}' for m in ['0.5', '0.75', '0.9091', '1.1', '1.3333', '2.0']))
        print('total dB  ' + ' '.join(f'{m}:{10*np.log10(sum(A[k][m] for k in ks)/nat):+.3f}' for m in ['0.5', '0.625', '0.75', '0.8', '0.9091', '1.1', '1.25', '1.3333', '1.6', '2.0']))
