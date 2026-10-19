# E2 independent reproduction (new code; original e2.py lost with old cloud container).
# Spec: 预注册-E2-E1-谱失配代价与多帧全扫-20261003.md §2 + 结果-E2 §1 (Hann-window metric). Choices not fixed by spec are listed in E2复现-实现选择-20261005.md
import os, json, time, argparse, glob, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
torch.set_num_threads(int(os.environ.get("NTH", "8"))); dev = "cpu"; HR = 256; L = 512
KAPPA = [1 / 16, 1 / 4, 4, 16]
def load(i):
    f = sorted(glob.glob('/workspace/SE-INR/Data/DIV2K_valid_HR/*.png'))[i]; im = np.asarray(Image.open(f).convert('RGB'), dtype=np.float64) / 255.
    h, w, _ = im.shape; y0, x0 = (h - HR) // 2, (w - HR) // 2; c = im[y0:y0 + HR, x0:x0 + HR]
    return os.path.basename(f)[:-4], torch.tensor(0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2], dtype=torch.float64)
def Chalf(u, sP): return torch.fft.ifft2(torch.fft.fft2(u) * sP).real
def gp_map_gen(A, y, Pspec, nv, iters=1500, tol=1e-6):
    sP = Pspec.sqrt()
    def At(v):
        z = torch.zeros(L, L, dtype=torch.float64, requires_grad=True)
        with torch.enable_grad(): g, = torch.autograd.grad(A(z), z, grad_outputs=v)
        return g
    def M(u): return Chalf(At(A(Chalf(u, sP))), sP) / nv + u
    rhs = Chalf(At(y), sP) / nv; u = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone(); rs = (r * r).sum()
    for _ in range(iters):
        Mp = M(p); al = rs / (p * Mp).sum(); u = u + al * p; r = r - al * Mp; rn = (r * r).sum()
        if rn.sqrt() < tol * rhs.norm(): break
        p = r + rn / rs * p; rs = rn
    return Chalf(u, sP)
fL = torch.fft.fftfreq(L, dtype=torch.float64); mL = torch.maximum(fL[:, None].abs(), fL[None].abs())
BL = [mL < 1 / 8, (mL >= 1 / 8) & (mL < 1 / 4), mL >= 1 / 4]
fH = np.fft.fftfreq(HR); mH = np.maximum(np.abs(fH)[:, None], np.abs(fH)[None]); BH = [mH < 1 / 8, (mH >= 1 / 8) & (mH < 1 / 4), mH >= 1 / 4]
hann = np.hanning(HR); W2 = hann[:, None] * hann[None]
def oracle_P(z):
    Z = torch.zeros(L, L, dtype=torch.float64); Z[:HR, :HR] = z - z.mean(); P = (torch.abs(torch.fft.fft2(Z)) ** 2 / (HR * HR)).numpy()
    return torch.tensor(gaussian_filter(P, 2, mode="wrap")) + 1e-12
def generic_P(var, alpha=2.0):
    P = (mL + 1.0 / L) ** (-alpha); P[0, 0] = 0; return P * var * L * L / P.sum() + 1e-12
def bandR(x, xh):
    X = np.abs(np.fft.fft2((x - x.mean()) * W2)) ** 2; E = np.abs(np.fft.fft2((x - xh) * W2)) ** 2
    return [float(1 - E[b].sum() / X[b].sum()) for b in BH]
def task_op(t):
    if t.startswith("single"):
        s = int(t[-1]); h = HR // s
        return (lambda z: F.interpolate(z[None, None, :HR, :HR], size=(h, h), mode='bicubic', antialias=True, align_corners=False)[0, 0]), 0.01
    rho = {"sparse10": .1, "sparse25": .25, "sparse50": .5}[t]; m = torch.rand(HR, HR, generator=torch.Generator().manual_seed(0)) < rho
    return (lambda z: z[:HR, :HR][m]), 0.002
ap = argparse.ArgumentParser(); ap.add_argument("--img", type=int, required=True); ap.add_argument("--task", required=True); ap.add_argument("--out", required=True); ap.add_argument("--iters", type=int, default=1500)
a = ap.parse_args(); t0 = time.time()
name, z = load(a.img); A, sig = task_op(a.task)
y = A(torch.cat([torch.cat([z, torch.zeros(HR, L - HR, dtype=torch.float64)], 1), torch.zeros(L - HR, L, dtype=torch.float64)], 0))
y = y + sig * torch.randn(y.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1)); mu = float(y.mean()); nv = sig ** 2
Pst = oracle_P(z); x = z.numpy()
def rec(P): return (gp_map_gen(A, y - mu, P, nv, a.iters)[:HR, :HR] + mu).numpy()
r = dict(img=name, task=a.task, matched=bandR(x, rec(Pst)), generic=bandR(x, rec(generic_P(float(z.var())))), mis={})
for b in range(3):
    for k in KAPPA:
        P = Pst.clone(); P[BL[b]] *= k; r["mis"][f"b{b+1}_{k}"] = bandR(x, rec(P))
r["sec"] = time.time() - t0; json.dump(r, open(a.out, "w")); print(name, a.task, "matched", [round(v, 3) for v in r["matched"]], f"{r['sec']:.0f}s", flush=True)
