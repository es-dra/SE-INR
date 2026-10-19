import sys, json, numpy as np, torch, argparse
from pathlib import Path
sys.path.insert(0, '/workspace/SE-INR/claude-20261019-cell-mismatch-v1/src'); import models
sys.argv += ['--src', 'x', '--ckpt', 'x', '--name', 'x']
import importlib.util
spec = importlib.util.spec_from_file_location('r3', '/workspace/SE-INR/claude-20261019-encoder-v3/run_v3.py'); R3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R3)
import torch.nn.functional as F
m = models.make({'name': 'edsr-baseline', 'args': {'n_resblocks': 32, 'n_feats': 256, 'res_scale': 0.1, 'scale': 4, 'rgb_range': 255}}); m.load_state_dict(torch.load('/workspace/SE-INR/SC-ASISR/artifacts/checkpoints/external/edsr/edsr_x4-4f62e9ef.pt', map_location='cpu')); m = m.cuda().eval()
rows = []; nrm = R3.N**4*3; s = 4
for p in R3.images('DIV2K-fresh', 0):
    x = R3.crop(p); T = R3.spec(x)
    y = F.interpolate(torch.from_numpy(x.transpose(2, 0, 1))[None], size=(R3.H//s,)*2, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    with torch.no_grad(): z = (m(y.cuda()*255)/255)[0].cpu().numpy().transpose(1, 2, 0)
    S = R3.spec(z)
    for lo, hi, mk in R3.masks(s):
        rows.append(dict(model='edsr-full', seed=0, s=s, lo=lo, img=Path(p).stem, err={'1.0': float((abs(T[mk]-S[mk])**2).sum()/nrm)}, tgt=float((abs(T[mk])**2).sum()/nrm)))
Path(sys.argv[1]).mkdir(); json.dump(rows, open(sys.argv[1]+'/rows.json', 'w')); print(len(rows))
