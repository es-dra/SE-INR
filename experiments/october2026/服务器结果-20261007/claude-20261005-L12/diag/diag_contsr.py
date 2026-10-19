# 事后诊断，描述性：ContinuousSR 在 ×2 附近（不改判定）
import sys, os
L = '/workspace/SE-INR/claude-20261005-L12'; sys.path.insert(0, L)
import torch, torch.nn.functional as F
from common import images, HR
CS = '/workspace/SE-INR/claude-20260920/ContinuousSR'; sys.path.insert(0, CS); os.chdir(CS)
import models
m = models.make(torch.load('/workspace/SE-INR/claude-20260920/contsr_w/ContinuousSR.pth', map_location='cpu', weights_only=False)['model'], load_sd=True).cuda().eval()
def mse(a, b): return float(((a.clamp(0, 1) - b) ** 2).mean())
imgs = list(images('Set14', 3))
print('B2: LR=round(240/s) (antialias bicubic), output 240x240, scale passed = 240/h')
for s in [1.9, 1.99, 2.0, 2.01, 2.1, 3.0, 4.0]:
    row = []
    for name, hr in imgs:
        h = round(HR / s); lr = F.interpolate(hr, size=(h, h), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
        with torch.no_grad(): o = m(lr.cuda(), torch.tensor([[HR / h, HR / h]], device='cuda'))[0].cpu()
        bic = F.interpolate(lr, size=(HR, HR), mode='bicubic', align_corners=False)[0]
        row.append(f'{name}: h={h} out={tuple(o.shape[-2:])} mse={mse(o, hr[0]):.5f} bic={mse(bic, hr[0]):.5f}')
    print(f' s={s}:', ' | '.join(row))
print('B2b: fixed LR h=120, scale exactly 2.0 vs 2.0+1e-6 vs output size via 2.0083 (=241/120)')
name, hr = imgs[0]; lr = F.interpolate(hr, size=(120, 120), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
for sc in [2.0, 2.000001, 1.999999]:
    with torch.no_grad(): o = m(lr.cuda(), torch.tensor([[sc, sc]], device='cuda'))[0].cpu()
    print(f'  scale={sc}: out={tuple(o.shape[-2:])} mse={mse(o[:, :HR, :HR], hr[0]):.5f} mean={float(o.mean()):.3f} min={float(o.min()):.2f} max={float(o.max()):.2f}')
print('B2c: exact x2 at other LR sizes (h=60->120, h=48->96), GT = antialias-downsampled HR')
for h in [60, 48]:
    gt = F.interpolate(hr, size=(2 * h, 2 * h), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
    lr = F.interpolate(hr, size=(h, h), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
    for sc in [2.0, 2.000001]:
        with torch.no_grad(): o = m(lr.cuda(), torch.tensor([[sc, sc]], device='cuda'))[0].cpu()
        print(f'  h={h} scale={sc}: out={tuple(o.shape[-2:])} mse={mse(o[:, :2*h, :2*h], gt[0]):.5f} max={float(o.max()):.2f}')
print("B2d: breakdown range scan (baboon), mean(out)/mean(HR)")
name, hr = imgs[0]
for s in [1.2, 1.5, 1.75, 2.0, 2.25, 2.5, 2.6, 2.75, 3.0]:
    h = round(HR / s); lr = F.interpolate(hr, size=(h, h), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    with torch.no_grad(): o = m(lr.cuda(), torch.tensor([[HR / h, HR / h]], device="cuda"))[0].cpu()
    print(f"  s={s} h={h} mse={mse(o, hr[0]):.5f} mean_ratio={float(o.mean()/hr.mean()):.3f}")
