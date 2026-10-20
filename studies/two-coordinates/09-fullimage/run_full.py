"""Full-image native-cell recovery per u-bin. Inference only."""
import sys, json, glob, hashlib, time, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--src'); ap.add_argument('--ckpt'); ap.add_argument('--name'); ap.add_argument('--needs-scale', action='store_true'); ap.add_argument('--out'); ap.add_argument('--bicubic', action='store_true')
a = ap.parse_args()
DATA = {'Set14': '/workspace/SE-INR/Data/Set14/HR', 'BSD100': '/workspace/SE-INR/Data/BSD100/HR', 'Urban100': '/workspace/SE-INR/Data/Urban100/HR'}
SC = (2, 3, 4, 6); SH = 4; dev = torch.device('cuda')
if not a.bicubic:
    sys.path.insert(0, a.src); import models
    c = torch.load(a.ckpt, map_location='cpu', weights_only=False); m = models.make(c['model'], load_sd=True).to(dev).eval()
acc = {}; t0 = time.time()
def bins(Hh, Ww, s):
    fy = np.fft.fftfreq(Hh); fx = np.fft.fftfreq(Ww); r = np.maximum(abs(fy)[:, None], abs(fx)[None, :]); u = 2*s*r
    return [(float(lo), (u >= lo) & (u < lo+.25) & (r > 0)) for lo in np.arange(0, s, .25)]
with torch.no_grad():
    for ds, root in DATA.items():
        for p in sorted(glob.glob(root+'/*.png')):
            im = np.asarray(Image.open(p).convert('RGB'), dtype='float32')/255
            Hh, Ww = (im.shape[0]//12)*12, (im.shape[1]//12)*12; x = im[:Hh, :Ww]
            T = np.fft.fft2(x[SH:Hh-SH, SH:Ww-SH].astype('float64'), axes=(0, 1)); nrm = (T.shape[0]*T.shape[1])**2*3
            xt = torch.from_numpy(x.transpose(2, 0, 1))[None]
            for s in SC:
                y = F.interpolate(xt, size=(Hh//s, Ww//s), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                if a.bicubic:
                    z = F.interpolate(y, size=(Hh, Ww), mode='bicubic', align_corners=False)[0].numpy().transpose(1, 2, 0)
                else:
                    m.gen_feat((y*2-1).to(dev))
                    vy = (np.arange(Hh)+.5)/Hh; vx = (np.arange(Ww)+.5)/Ww
                    q = torch.from_numpy((2*np.stack(np.meshgrid(vy, vx, indexing='ij'), -1).reshape(-1, 2)-1).astype('float32'))[None].to(dev)
                    cell = torch.ones_like(q); cell[..., 0] *= 2/Hh; cell[..., 1] *= 2/Ww
                    if a.needs_scale:
                        z = m.query_rgb(q, torch.full((1,), float(s), device=dev), cell)[0]
                    else:
                        z = torch.cat([m.query_rgb(q[:, i:i+60000], cell[:, i:i+60000])[0] for i in range(0, q.shape[1], 60000)])
                    z = (z.float().cpu().numpy()*.5+.5).reshape(Hh, Ww, 3)
                S = np.fft.fft2(z[SH:Hh-SH, SH:Ww-SH].astype('float64'), axes=(0, 1))
                for lo, mk in bins(Hh-2*SH, Ww-2*SH, s):
                    k = f'{ds}|{s}|{lo}'; e = acc.setdefault(k, [0.0, 0.0])
                    e[0] += float((abs(T[mk]-S[mk])**2).sum()/nrm); e[1] += float((abs(T[mk])**2).sum()/nrm)
            print(a.name, ds, Path(p).stem, round(time.time()-t0, 1), flush=True)
json.dump(dict(model=a.name, ckpt=a.ckpt, acc=acc, seconds=time.time()-t0), open(a.out, 'w'))
