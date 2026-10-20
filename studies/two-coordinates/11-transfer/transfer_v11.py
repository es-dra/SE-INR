import sys, glob, json, collections, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/SE-INR/runs/claude-20261019-cell-mismatch-v1/src'); import models
from PIL import Image
H = 240; SH = 4; N = H-2*SH; f = np.fft.fftfreq(N); RR = np.maximum(abs(f[:, None]), abs(f[None, :])); ib = (RR/0.03125).astype(int)
def spec(x): return np.fft.fft2(x.astype('float64')[SH:H-SH, SH:H-SH], axes=(0, 1))
dev = torch.device('cuda'); MS = ['1.0', '0.5', '1.3333', '2.0']
v = (np.arange(H)+.5)/H; q = torch.from_numpy((2*np.stack(np.meshgrid(v, v, indexing='ij'), -1).reshape(-1, 2)-1).astype('float32'))[None].to(dev)
A = collections.defaultdict(lambda: np.zeros(6))  # <e,P1>, |P1|^2, |T|^2, |e|^2, <P1,Pm>, dMSE
with torch.no_grad():
    for mo in ['liif', 'lte', 'sc-inr']:
        c = torch.load(f'/workspace/SE-INR/SC-ASISR/artifacts/checkpoints/seed1/{mo}/epoch-last.pth', map_location='cpu', weights_only=False)
        m = models.make(c['model'], load_sd=True).to(dev).eval()
        for p in sorted(glob.glob('/workspace/SE-INR/Data/DIV2K_valid_HR/*.png')):
            if not 869 <= int(p[-8:-4]) <= 900: continue
            im = Image.open(p).convert('RGB'); w, h = im.size; l, t = (w-H)//2, (h-H)//2
            x = np.asarray(im.crop((l, t, l+H, t+H)), dtype='float32')/255; T = spec(x); xt = torch.from_numpy(x.transpose(2, 0, 1))[None]
            for s in (2, 3, 4, 6):
                y = F.interpolate(xt, size=(H//s,)*2, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                m.gen_feat((y*2-1).to(dev))
                S = {mu: spec(m.query_rgb(q, torch.full_like(q, float(mu)*2/H))[0].float().cpu().numpy().reshape(H, H, 3)*.5+.5) for mu in MS}
                P1 = S['1.0']
                for b in range(16):
                    mk = (ib == b) & (RR > 0); p1 = P1[mk]; tt = T[mk]; e = tt-p1
                    for mu in MS[1:]:
                        pm = S[mu][mk]
                        A[(mo, s, mu, b)] += [(np.conj(e)*p1).real.sum(), (abs(p1)**2).sum(), (abs(tt)**2).sum(), (abs(e)**2).sum(), (np.conj(p1)*pm).real.sum(), (abs(tt-pm)**2).sum()-(abs(e)**2).sum()]
            print(mo, p[-8:-4], flush=True)
json.dump({str(k): v.tolist() for k, v in A.items()}, open('transfer_v11.json', 'w'))
res = {'W1': [], 'W2': [], 'W3': []}
for mo in ['liif', 'lte', 'sc-inr']:
    for s in (2, 3, 4):
        ws = {b: A[(mo, s, '2.0', b)][4]/A[(mo, s, '2.0', b)][1] for b in range(16) if A[(mo, s, '2.0', b)][1] > 0}
        bmin = min(ws, key=ws.get); res['W1'].append(dict(model=mo, s=s, r_min=bmin/32, ok=bool(0.25 <= bmin/32 < 0.375)))
        for b in range(16):
            vv = A[(mo, s, '2.0', b)]
            if vv[1] > 0 and 2*s*b/32 < 2: res['W3'].append(bool(abs(vv[0])/vv[1] <= .15))
        if s in (3, 4):
            act = sum(A[(mo, s, '2.0', b)][5] for b in range(16))
            pred = sum((1-A[(mo, s, '2.0', b)][4]/A[(mo, s, '2.0', b)][1])**2*(A[(mo, s, '2.0', b)][2]-A[(mo, s, '2.0', b)][3]) for b in range(16) if A[(mo, s, '2.0', b)][1] > 0)
            res['W2'].append(dict(model=mo, s=s, ratio=pred/act, ok=bool(.7 <= pred/act <= 1.3)))
summ = dict(W1=f"{sum(x['ok'] for x in res['W1'])}/9", W1_PASS=all(x['ok'] for x in res['W1']), W2=f"{sum(x['ok'] for x in res['W2'])}/6", W2_PASS=sum(x['ok'] for x in res['W2']) >= 5,
            W3=f"{sum(res['W3'])}/{len(res['W3'])}", W3_PASS=sum(res['W3'])/len(res['W3']) >= .9, detail=res)
json.dump(summ, open('score_v11.json', 'w'), indent=1, default=str); print(json.dumps({k: v for k, v in summ.items() if k != 'detail'}, indent=1), json.dumps(res['W2'], default=str), json.dumps(res['W1'], default=str))
