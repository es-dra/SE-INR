import sys, glob, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, 'src'); from datasets.gauss_wrapper import gauss_down
H = 240; SH = 4; N = H-2*SH; f = np.fft.fftfreq(N); RR = np.maximum(abs(f[:, None]), abs(f[None, :]))
spec = lambda x: np.fft.fft2(x.astype('float64')[SH:H-SH, SH:H-SH], axes=(0, 1))
E_ = {}; T_ = {}
for p in sorted(glob.glob('/workspace/SE-INR/Data/DIV2K_valid_HR/*.png')):
    if not 869 <= int(p[-8:-4]) <= 900: continue
    im = Image.open(p).convert('RGB'); w, h = im.size; l, t = (w-H)//2, (h-H)//2
    x = np.asarray(im.crop((l, t, l+H, t+H)), dtype='float32')/255; T = spec(x); xt = torch.from_numpy(x.transpose(2, 0, 1))
    for s in (2, 3, 4, 6):
        y = gauss_down(xt, (H//s, H//s))[None]; z = F.interpolate(y, size=(H, H), mode='bicubic', align_corners=False)[0].numpy().transpose(1, 2, 0); S = spec(z)
        for lo in np.arange(0, s, .25):
            mk = (2*s*RR >= lo) & (2*s*RR < lo+.25) & (RR > 0); k = f'{s}|{lo}'
            E_[k] = E_.get(k, 0)+(abs(T[mk]-S[mk])**2).sum(); T_[k] = T_.get(k, 0)+(abs(T[mk])**2).sum()
json.dump({k: float(1-E_[k]/T_[k]) for k in E_}, open('bicubic_R.json', 'w'), indent=1); print('ok')
