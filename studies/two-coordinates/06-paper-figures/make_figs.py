"""Paper figures from frozen run outputs (read-only). Writes fig1_recovery.pdf, fig2_cell.pdf, numbers.json."""
import json, collections, glob, sys, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
RUNS = '/workspace/SE-INR/runs'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'lines.linewidth': 1.2})
COL = {2: '#1b6ca8', 3: '#2a9d8f', 4: '#e9a03b', 6: '#c8553d'}
def load(p, key=('model', 'seed', 's', 'lo')):
    A = collections.defaultdict(collections.Counter); T = collections.Counter()
    for r in json.load(open(p)):
        k = tuple(r[x] for x in key); A[k].update(r['err']); T[k] += r['tgt']
    return A, T
A2, T2 = load(f'{RUNS}/claude-20261019-two-coordinates-v2/formal/rows.json', ('model', 's', 'lo'))
R = lambda A, T, k: 1 - A[k]['1.0']/T[k]
# bicubic R_lin on the same cohort
H = 240; SH = 4; N = H-2*SH; f = np.fft.fftfreq(N); RR = np.maximum(abs(f[:, None]), abs(f[None, :]))
def spec(x): return np.fft.fft2(x.astype('float64')[SH:H-SH, SH:H-SH], axes=(0, 1))
BL = collections.Counter(); BT = collections.Counter()
for p in sorted(glob.glob('/workspace/SE-INR/Data/DIV2K_valid_HR/*.png')):
    if not 869 <= int(p[-8:-4]) <= 900: continue
    im = Image.open(p).convert('RGB'); w, h = im.size; l, t = (w-H)//2, (h-H)//2
    x = np.asarray(im.crop((l, t, l+H, t+H)), dtype='float32')/255; T = spec(x); xt = torch.from_numpy(x.transpose(2, 0, 1))[None]
    for s in (2, 3, 4, 6):
        y = F.interpolate(xt, size=(H//s,)*2, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        up = F.interpolate(y, size=(H, H), mode='bicubic', align_corners=False)[0].numpy().transpose(1, 2, 0); S = spec(up)
        u = 2*s*RR
        for lo in np.arange(0, s, .25):
            mk = (u >= lo) & (u < lo+.25) & (RR > 0); BL[(s, lo)] += (abs(T[mk]-S[mk])**2).sum(); BT[(s, lo)] += (abs(T[mk])**2).sum()
num = {}
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
for s in (2, 3, 4, 6):
    us = np.arange(0, s, .25)
    rn = [np.mean([R(A2, T2, (m, s, float(b))) for m in ['liif', 'lte', 'sc-inr']]) for b in us]
    rl = [1 - BL[(s, b)]/BT[(s, b)] for b in us]
    ax[0].plot(us+.125, rn, '-', color=COL[s], label=f'×{s} network'); ax[0].plot(us+.125, rl, ':', color=COL[s])
    num[f'Rnet_x{s}'] = dict(zip([f'{b:.2f}' for b in us], map(float, rn))); num[f'Rlin_x{s}'] = dict(zip([f'{b:.2f}' for b in us], map(float, rl)))
ax[0].axvline(1, color='k', lw=.6, ls='--'); ax[0].set_xlim(0, 4); ax[0].set_xlabel('u = ν / ν_LR'); ax[0].set_ylabel('band recovery R(u)')
ax[0].set_title('(a) recovery vs u'); ax[0].legend(frameon=False, loc='upper right')
# (b) decoder spread at x4 incl. GaussianSR
Ag, Tg = load(f'{RUNS}/claude-20261019-encoder-v3/formal_gs/rows.json', ('model', 's', 'lo'))
us = np.arange(0, 4, .25)
for m, lab, c in [('liif', 'LIIF', '#1b6ca8'), ('lte', 'LTE', '#2a9d8f'), ('sc-inr', 'SC-INR', '#e9a03b')]:
    ax[1].plot(us+.125, [R(A2, T2, (m, 4, float(b))) for b in us], color=c, label=lab)
ax[1].plot(us+.125, [R(Ag, Tg, ('gaussiansr', 4, float(b))) for b in us], color='#c8553d', ls='--', label='GaussianSR')
ax[1].axvline(1, color='k', lw=.6, ls='--'); ax[1].set_xlabel('u'); ax[1].set_title('(b) four decoders, ×4'); ax[1].legend(frameon=False)
# (c) what changes R(u): all at x4, differences in R
V5 = '/workspace/SE-INR/runs/claude-20261019-encoder-train-v5'; V7 = '/workspace/SE-INR/runs/claude-20261020-swinir-v7'
A1, T1 = load(f'{RUNS}/claude-20261019-two-coordinates-v2/formal/rows.json')
EA = {m: load(f'{RUNS}/claude-20261019-prior-v4/early_{m}/rows.json', ('model', 's', 'lo')) for m in ['liif', 'lte', 'sc-inr']}
us = np.arange(0, 4, .25); s = 4
d = [np.mean([R(A1, T1, (m, 1, s, float(b))) - (1-EA[m][0][(m, s, float(b))]['1.0']/EA[m][1][(m, s, float(b))]) for m in ['liif', 'lte', 'sc-inr']]) for b in us]
ax[2].plot(us+.125, d, color='#555555', label='training 100→1000 ep'); num['dR_training_x4'] = list(map(float, d))
def pair(pa, pb, lab, c, ls):
    Aa, Ta = load(pa, ('s', 'lo')); Ab, Tb = load(pb, ('s', 'lo'))
    v = [R(Aa, Ta, (s, float(b))) - R(Ab, Tb, (s, float(b))) for b in us]; num['dR_'+lab] = list(map(float, v))
    ax[2].plot(us+.125, v, color=c, ls=ls, label=lab)
pair(f'{V5}/eval_rdn/rows.json', f'{V5}/eval_edsr/rows.json', 'RDN − EDSR encoder', '#1b6ca8', '-')
pair(f'{V7}/eval200_swinir/rows.json', f'{V5}/eval_edsr/rows.json', 'SwinIR − EDSR encoder', '#2a9d8f', '--')
pair(f'{V5}/eval_tinydec/rows.json', f'{V5}/eval_edsr/rows.json', '2×64 − 4×256 decoder', '#c8553d', ':')
ax[2].axvline(1, color='k', lw=.6, ls='--'); ax[2].axhline(0, color='k', lw=.4); ax[2].set_xlabel('u'); ax[2].set_ylabel('ΔR (×4)'); ax[2].set_title('(c) what changes R(u)'); ax[2].legend(frameon=False, fontsize=5.5, loc='lower left')
fig.tight_layout(); fig.savefig('fig1_recovery.pdf'); fig.savefig('fig1_recovery.png', dpi=200)
# Fig 2
tr = {tuple(eval(k)): np.array(v) for k, v in json.load(open('/workspace/SE-INR/runs/claude-20261020-transfer-v11/transfer_v11.json')).items()}
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
for s in (2, 3, 4, 6):
    rb = [b for b in range(16) if all(tr[(m, s, '2.0', b)][1] > 0 for m in ['liif', 'lte', 'sc-inr'])]
    ws = [np.mean([tr[(m, s, '2.0', b)][4]/tr[(m, s, '2.0', b)][1] for m in ['liif', 'lte', 'sc-inr']]) for b in rb]
    ax[0].plot([b/32+1/64 for b in rb], ws, color=COL[s], label=f'×{s}')
ax[0].set_xlabel('r (cycles / HR pixel)'); ax[0].set_ylabel('w(r)'); ax[0].set_title('(a) transfer of cell×2'); ax[0].legend(frameon=False)
A2s, T2s = load(f'{RUNS}/claude-20261019-two-coordinates-v2/formal/rows.json', ('model', 's', 'lo'))
for s in (2, 3, 4, 6):
    nat = sum(A2s[k]['1.0'] for k in A2s if k[0] == 'lte' and k[1] == s); Hh = collections.Counter()
    for b in np.arange(0, s, .25): k = ('lte', s, float(b)); Hh[int(b/(2*s)/0.0625+1e-9)] += 100*(A2s[k]['2.0']-A2s[k]['1.0'])/nat
    ax[1].plot([i*0.0625+0.03125 for i in range(8)], [Hh[i] for i in range(8)], 'o-', ms=2.5, color=COL[s], label=f'×{s}')
ax[1].set_xlabel('r (cycles / HR pixel)'); ax[1].set_ylabel('cost of cell×2 (% native MSE)'); ax[1].set_title('(b) where the cost lands (LTE)'); ax[1].legend(frameon=False)
cd = {}
for m, c in [('liif', '#1b6ca8'), ('lte', '#2a9d8f'), ('sc-inr', '#e9a03b')]:
    y = []
    for s in (2, 3, 4, 6):
        n = sum(A2s[k]['1.0'] for k in A2s if k[0] == m and k[1] == s); d2 = sum(A2s[k]['2.0'] for k in A2s if k[0] == m and k[1] == s)
        h = sum(A2s[k]['0.5'] for k in A2s if k[0] == m and k[1] == s); y.append((10*np.log10(d2/n), 10*np.log10(h/n)))
    cd[m] = y
    ax[2].plot([2, 3, 4, 6], [a for a, b in y], 'o-', ms=2.5, color=c, label=f'{m.upper()} cell×2')
    ax[2].plot([2, 3, 4, 6], [b for a, b in y], 's:', ms=2.5, color=c)
ax[2].set_xlabel('scale'); ax[2].set_ylabel('PSNR loss (dB)'); ax[2].set_title('(c) cost vs scale'); ax[2].legend(frameon=False)
num['cell_cost_db'] = cd
fig.tight_layout(); fig.savefig('fig2_cell.pdf'); fig.savefig('fig2_cell.png', dpi=200)
json.dump(num, open('numbers.json', 'w'), indent=1); print('ok')
