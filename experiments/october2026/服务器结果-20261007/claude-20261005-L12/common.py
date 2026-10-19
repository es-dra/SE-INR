# L1/L2 shared helpers (prereg 预注册-L1L2-20261005.md). No model imports here.
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
HR = 240; NK = HR // 2 + 1          # Chebyshev radius index k = nu*HR, k = 0..120
DATA = Path('/workspace/SE-INR/Data')
CK = Path('/workspace/SE-INR/SC-ASISR/artifacts/checkpoints')
_f = np.fft.fftfreq(HR); KIDX = np.rint(np.maximum(np.abs(_f)[:, None], np.abs(_f)[None]) * HR).astype(int)

def images(ds, limit=30):
    for fp in sorted((DATA / ds / 'HR').glob('*.png'))[:limit]:
        im = Image.open(fp).convert('RGB'); w, h = im.size
        if min(w, h) < HR: continue
        x0, y0 = (w - HR) // 2, (h - HR) // 2
        yield fp.name, transforms.ToTensor()(im.crop((x0, y0, x0 + HR, y0 + HR)))[None]   # 1x3xHRxHR in [0,1]

def degrade(hr, s):
    return F.interpolate(hr, size=(HR // s, HR // s), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)

def spec_S(z):
    """z: HRxHR grey (np float64). Per-k signal energy of demeaned z."""
    X = np.fft.fft2(z - z.mean())
    return np.bincount(KIDX.ravel(), (np.abs(X) ** 2).ravel(), minlength=NK)

def spec_E(z, zh):
    E = np.fft.fft2(z - zh)
    return np.bincount(KIDX.ravel(), (np.abs(E) ** 2).ravel(), minlength=NK)

def grey(t):
    """t: 3xHxW or 1x3xHxW torch -> HxW float64 numpy (RGB mean)."""
    t = t.detach().float().cpu()
    if t.dim() == 4: t = t[0]
    return t.mean(0).double().numpy()
