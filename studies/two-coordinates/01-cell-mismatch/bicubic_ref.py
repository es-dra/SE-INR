"""Bicubic-upsampling per-u-bin recovery on the same crops/degradation as run.py (linear reference). CPU only."""
import json, sys, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.'); import run as R0
out = sys.argv[1]; rows = []; nrm = R0.N**4*3
for ds in R0.DATA:
    for p in R0.images(ds, 0):
        x = R0.crop(p); T = R0.spec(x); xt = torch.from_numpy(x.transpose(2, 0, 1))[None]
        for s in [2, 3, 4, 6]:
            y = F.interpolate(xt, size=(R0.H//s,)*2, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
            up = F.interpolate(y, size=(R0.H,)*2, mode='bicubic', align_corners=False)[0].numpy().transpose(1, 2, 0)
            S = R0.spec(up)
            for lo, hi, mk in R0.masks(s):
                rows.append(dict(ds=ds, img=p.split('/')[-1][:-4], s=s, lo=lo, err=float((abs(T[mk]-S[mk])**2).sum()/nrm), tgt=float((abs(T[mk])**2).sum()/nrm)))
json.dump(rows, open(out, 'w')); print(len(rows))
