"""Qualitative x4 comparison. Selection rule fixed before looking at outputs:
image = Urban100 image whose RDN-minus-EDSR PSNR gain (Y, whole image) is the median over Urban100;
crop  = 48x48 HR window (stride 8) with the largest HR gradient energy in that image."""
import sys, glob, json, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
ROOT = '/workspace/SE-INR/SE-INR-current'; sys.path.insert(0, ROOT)
import models
from datasets.wrappers import resize_fn
from torchvision import transforms
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
dev = torch.device('cuda'); S = 4
V5 = '/workspace/SE-INR/runs/claude-20261019-encoder-train-v5/save'; V7 = '/workspace/SE-INR/runs/claude-20261020-swinir-v7/save'
CK = {'EDSR-LIIF': f'{V5}/liif-edsr-200ep/epoch-last.pth', 'no cell': f'{V5}/liif-edsr-nocell-200ep/epoch-last.pth',
      'decoder 2x64': f'{V5}/liif-edsr-tinydec-200ep/epoch-last.pth', 'RDN-LIIF': f'{V5}/liif-rdn-200ep/epoch-last.pth',
      'SwinIR-LIIF': f'{V7}/liif-swinir-200ep/epoch-last.pth'}
def y(x): return (16 + 65.481*x[..., 0] + 128.553*x[..., 1] + 24.966*x[..., 2])/255
def psnr(a, b, sh=S):
    d = (y(a) - y(b))[sh:-sh, sh:-sh]; return -10*np.log10((d**2).mean())
@torch.no_grad()
def sr(m, lr, H, W):
    m.gen_feat((lr*2-1).to(dev)); m.feat = m.feat[:, :, :lr.shape[-2], :lr.shape[-1]]
    vy, vx = (np.arange(H)+.5)/H, (np.arange(W)+.5)/W
    q = torch.from_numpy((2*np.stack(np.meshgrid(vy, vx, indexing='ij'), -1).reshape(-1, 2)-1).astype('float32'))[None].to(dev)
    c = torch.ones_like(q); c[..., 0] *= 2/H; c[..., 1] *= 2/W
    z = torch.cat([m.query_rgb(q[:, i:i+60000], c[:, i:i+60000])[0] for i in range(0, q.shape[1], 60000)])
    return (z.float().cpu().numpy()*.5+.5).clip(0, 1).reshape(H, W, 3)
def load(p):
    ck = torch.load(p, map_location='cpu', weights_only=False); return models.make(ck['model'], load_sd=True).to(dev).eval()
# 1) pick image by median RDN gain
res = {}
for l in open('/workspace/SE-INR/runs/claude-20261022-benchmark-v13/results.txt'): pass
files = sorted(glob.glob('/workspace/SE-INR/Data/Urban100/HR/*.png'))
imgs = {}
for p in files:
    hr = np.asarray(Image.open(p).convert('RGB'), dtype='float32')/255; H, W = (hr.shape[0]//S)*S, (hr.shape[1]//S)*S; hr = hr[:H, :W]
    lr = resize_fn(torch.from_numpy(hr.transpose(2, 0, 1)), (H//S, W//S))[None]
    imgs[p] = (hr, lr)
mE, mR = load(CK['EDSR-LIIF']), load(CK['RDN-LIIF'])
gain = {p: psnr(sr(mR, lr, *hr.shape[:2]), hr) - psnr(sr(mE, lr, *hr.shape[:2]), hr) for p, (hr, lr) in imgs.items()}
order = sorted(gain, key=gain.get); pick = order[len(order)//2]
hr, lr = imgs[pick]; H, W = hr.shape[:2]
# 2) crop with max gradient energy
g = y(hr); e = np.abs(np.diff(g, axis=0))[:, :-1]**2 + np.abs(np.diff(g, axis=1))[:-1, :]**2
best, pos = -1, (0, 0); C = 48
for i in range(0, H-C-1, 8):
    for j in range(0, W-C-1, 8):
        v = e[i:i+C, j:j+C].sum()
        if v > best: best, pos = v, (i, j)
i, j = pos
outs = {'HR': hr}
outs['Bicubic'] = F.interpolate(lr, size=(H, W), mode='bicubic', align_corners=False)[0].numpy().transpose(1, 2, 0).clip(0, 1)
for k, p in CK.items():
    m = load(p); outs[k] = sr(m, lr, H, W); del m; torch.cuda.empty_cache()
meta = dict(image=pick.split('/')[-1], median_gain_db=gain[pick], crop=[i, j, C], psnr_full={k: (None if k == 'HR' else round(psnr(v, hr), 2)) for k, v in outs.items()},
            psnr_crop={k: (None if k == 'HR' else round(-10*np.log10(((y(v)-y(hr))[i:i+C, j:j+C]**2).mean()), 2)) for k, v in outs.items()})
json.dump(meta, open('qual_meta.json', 'w'), indent=1); print(json.dumps(meta, indent=1))
np.savez_compressed('crops.npz', **{k: v[i:i+C, j:j+C] for k, v in outs.items()})
order_k = ['HR', 'Bicubic', 'EDSR-LIIF', 'no cell', 'decoder 2x64', 'RDN-LIIF', 'SwinIR-LIIF']
fig, ax = plt.subplots(2, 7, figsize=(7.0, 2.35))
for c, k in enumerate(order_k):
    ax[0, c].imshow(outs[k][i:i+C, j:j+C], interpolation='nearest')
    ax[0, c].set_title(k if k == 'HR' else f'{k}\n{meta["psnr_crop"][k]:.2f} dB', fontsize=6.5)
    if k == 'HR':
        ax[1, c].text(0.5, 0.5, '|error| (Y)\n0 to 0.25', ha='center', va='center', fontsize=6.5, transform=ax[1, c].transAxes)
    else:
        ax[1, c].imshow(np.abs(y(outs[k]) - y(hr))[i:i+C, j:j+C], cmap='magma', vmin=0, vmax=0.25, interpolation='nearest')
    for r in (0, 1): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
ax[1, 0].axis('off')
fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.04, hspace=0.04)
fig.savefig('fig3_qualitative.pdf'); fig.savefig('fig3_qualitative.png', dpi=250)
