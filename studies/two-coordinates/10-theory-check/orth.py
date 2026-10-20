"""Exploratory check of the leverage law's assumptions on saved Set14 predictions (v1):
(1) orthogonality <e,P1>_b ~ 0  => emitted energy |P1_b|^2 ~ recovered energy R_b E_b
(2) predicted cell cost sum_b (1-w_b)^2 |P1_b|^2 vs actual, per model x scale x m."""
import glob, json, collections, numpy as np
from PIL import Image
H = 240; SH = 4; N = H-2*SH; f = np.fft.fftfreq(N); RR = np.maximum(abs(f[:, None]), abs(f[None, :]))
def spec(x): return np.fft.fft2(x.astype('float64')[SH:H-SH, SH:H-SH], axes=(0, 1))
def crop(p):
    im = Image.open(p).convert('RGB'); w, h = im.size; l, t = (w-H)//2, (h-H)//2
    return np.asarray(im.crop((l, t, l+H, t+H)), dtype='float32')/255
ib = (RR/0.03125).astype(int)
A = collections.defaultdict(lambda: np.zeros(8))
for p in sorted(glob.glob('/workspace/SE-INR/runs/claude-20261019-cell-mismatch-v1/formal/pred/*.npz')):
    ds, rest = p.split('/')[-1][:-4].split('_', 1); stem, s, model, seed = rest.rsplit('_', 3); s = int(s[1:])
    T = spec(crop(f'/workspace/SE-INR/Data/Set14/HR/{stem}.png')); z = np.load(p); P1 = spec(z['1.0'].astype('float32'))
    for m in ['2.0', '1.3333', '0.5']:
        Pm = spec(z[m].astype('float32'))
        for b in range(16):
            mk = (ib == b) & (RR > 0); p1 = P1[mk]; pm = Pm[mk]; t = T[mk]; e = t-p1
            A[(model, s, m, b)] += [ (np.conj(e)*p1).real.sum(), (abs(p1)**2).sum(), (abs(t)**2).sum(), (abs(e)**2).sum(),
                                     (np.conj(p1)*pm).real.sum(), (abs(t-pm)**2).sum()-(abs(e)**2).sum(), 0, 0]
out = {}
for model in ['liif', 'lte', 'sc-inr']:
    for s in (2, 3, 4, 6):
        o = []
        for b in range(16):
            v = A[(model, s, '2.0', b)]
            if v[1] == 0: continue
            o.append((v[0]/v[1], v[1]/(v[2]-v[3]) if v[2]-v[3] > 0 else np.nan))
        out[f'{model}/x{s}/orth_ratio_and_emitted_over_recovered'] = [[round(a, 3), round(b, 3)] for a, b in o]
        for m in ['2.0', '1.3333', '0.5']:
            act = sum(A[(model, s, m, b)][5] for b in range(16)); nat = sum(A[(model, s, m, b)][3] for b in range(16))
            pred_emit = sum((1-A[(model, s, m, b)][4]/A[(model, s, m, b)][1])**2*A[(model, s, m, b)][1] for b in range(16) if A[(model, s, m, b)][1] > 0)
            pred_rec = sum((1-A[(model, s, m, b)][4]/A[(model, s, m, b)][1])**2*(A[(model, s, m, b)][2]-A[(model, s, m, b)][3]) for b in range(16) if A[(model, s, m, b)][1] > 0)
            out[f'{model}/x{s}/m{m}'] = dict(actual_pct=round(100*act/nat, 3), pred_emitted_pct=round(100*pred_emit/nat, 3), pred_recovered_pct=round(100*pred_rec/nat, 3))
json.dump(out, open('orth.json', 'w'), indent=1)
for k, v in out.items():
    if '/m' in k: print(k, v)
for k in ['lte/x2/orth_ratio_and_emitted_over_recovered', 'lte/x4/orth_ratio_and_emitted_over_recovered']: print(k, out[k])
