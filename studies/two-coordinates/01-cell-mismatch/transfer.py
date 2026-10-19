"""Exploratory: is the cell's effect a linear spectral filter w(r) on the native output? Uses v1 saved Set14 preds (float16)."""
import glob, json, collections, numpy as np, sys
sys.path.insert(0, '.')
from PIL import Image
H = 240; SH = 4; N = H-2*SH
f = np.fft.fftfreq(N); R = np.maximum(abs(f[:, None]), abs(f[None, :]))
def spec(x): return np.fft.fft2(x.astype('float64')[SH:H-SH, SH:H-SH], axes=(0, 1))
def crop(p):
    im = Image.open(p).convert('RGB'); w, h = im.size; l, t = (w-H)//2, (h-H)//2
    return np.asarray(im.crop((l, t, l+H, t+H)), dtype='float32')/255
acc = collections.defaultdict(lambda: np.zeros(5))
for p in sorted(glob.glob('formal/pred/*.npz')):
    ds, rest = p.split('/')[-1][:-4].split('_', 1)
    stem, s, model, seed = rest.rsplit('_', 3); s = int(s[1:])
    T = spec(crop(f'/workspace/SE-INR/Data/Set14/HR/{stem}.png')); z = np.load(p); P1 = spec(z['1.0'])
    for m in ['0.5', '0.75', '1.3333', '2.0']:
        Pm = spec(z[m]); ib = (R/0.03125).astype(int)  # r-bins width 1/32
        for b in range(16):
            mk = (ib == b) & (R > 0)
            if not mk.any(): continue
            p1 = P1[mk]; pm = Pm[mk]; t = T[mk]
            w = (np.conj(p1)*pm).real.sum()/(abs(p1)**2).sum()
            resid = abs(pm - w*p1)**2
            acc[(model, s, m, b)] += [w*(abs(p1)**2).sum(), (abs(p1)**2).sum(), resid.sum(), (abs(t-pm)**2).sum()-(abs(t-p1)**2).sum(), (abs(pm-p1)**2).sum()]
out = {str(k): v.tolist() for k, v in acc.items()}; json.dump(out, open('transfer.json', 'w'))
for model in ['lte']:
    for m in ['2.0', '0.5']:
        print(f'{model} m={m}: w(r) per r-bin [r start] (fraction of change not explained by scalar filter)')
        for s in [2, 3, 4, 6]:
            row = []
            for b in range(16):
                v = acc[(model, s, m, b)]
                if v[1] == 0: continue
                row.append(f'{b/32:.2f}:{v[0]/v[1]:.2f}({v[2]/max(v[4],1e-30):.2f})')
            print(f'  x{s} ' + ' '.join(row))
