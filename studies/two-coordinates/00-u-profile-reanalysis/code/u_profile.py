"""Read-only re-analysis of cell-reference-formal-v2 saved predictions in LR-Nyquist-normalised frequency u=2*s*r.
No inference, no training. Writes only to --out."""
import json, sys, glob, os, numpy as np
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=False)
EDGES = [0, .25, .5, .75, 1, 1.25, 1.5, 2, 3, 4.0001]
CELLS = {'zero': 0.0, 'h4': .5, 'h3': 2/3, 'h2': 1.0}
NATIVE = {2: 'h2', 4: 'h4'}
f = np.fft.fftfreq(120); R = np.maximum(abs(f[:, None]), abs(f[None, :]))
def spec(x):
    return np.fft.fft2(x.astype('float64')[4:124, 4:124], axes=(0, 1))
rows = []
for p in sorted(glob.glob(f'{src}/predictions/*.npz')):
    name = os.path.basename(p)[:-4]; img, rest = name.split('_x'); s, model = rest.split('__'); s = int(s); img = int(img)
    T = spec(np.load(f'{src}/truth/{img}.npy'))
    z = np.load(p); P = {k: spec(z[k]) for k in z.files}
    u = 2 * s * R
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (u >= lo) & (u < hi) & (R > 0)
        if not m.any(): continue
        n = 120 * 120 * 3 * 120 * 120  # Parseval: mean over pixels = sum|X|^2 / N^2 per channel
        err = {k: float((abs(T - P[k])[m] ** 2).sum() / n) for k in P}
        for ref in P:
            if ref == NATIVE[s]: continue
            d = P[NATIVE[s]] - P[ref]; e = T - P[ref]
            a = float((e.conj() * d).real[m].sum() / n); b = float((abs(d)[m] ** 2).sum() / n)
            rows.append(dict(img=img, s=s, model=model.split('_')[0], seed=int(model[-1]), ref=ref, lo=lo, hi=hi,
                             e_ref=err[ref], e_nat=err[NATIVE[s]], a=a, b=b, gain=err[ref] - err[NATIVE[s]],
                             err_all=err, tgt=float((abs(T)[m] ** 2).sum() / n)))
json.dump(rows, open(f'{out}/rows.json', 'w'))
print('rows', len(rows))
