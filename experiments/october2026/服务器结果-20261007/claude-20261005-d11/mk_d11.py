# build d11gp.py from e1gp.py: same functions, device=cpu, new main loop that records band energies (prereg D11)
src = open("/workspace/SE-INR/claude-20261004-e1/e1gp.py").read()
src = src.replace("dev = 'cuda'", "dev = 'cpu'", 1)
i = src.index("ap = argparse.ArgumentParser()")
main = r'''
import os
torch.set_num_threads(int(os.environ.get("NTH", "8")))
def bands_HR(n):  # cycles/px on n-grid
    f = torch.fft.fftfreq(n); m = torch.maximum(f[:, None].abs(), f[None].abs())
    return [(m < 1 / 8), (m >= 1 / 8) & (m < 1 / 4), (m >= 1 / 4)]
BW = bands_HR(240); BL = [b.to(dev) for b in bands_HR(L)]
hann = torch.hann_window(240, periodic=False); W2 = hann[:, None] * hann[None]
def band_energy(z):  # z (3,256,256) -> per band windowed energy summed over channels
    c = z[:, 8:-8, 8:-8] * W2; S = (torch.fft.fft2(c).abs() ** 2).sum(0)
    return [float(S[b].sum()) for b in BW]
def band_P(P): return [float(P[b].sum()) for b in BL]
import numpy as np
def spec(z): c = z[:, 8:-8, 8:-8] * W2; return (torch.fft.fft2(c).abs() ** 2).sum(0).numpy().astype(np.float32)
ap = argparse.ArgumentParser(); ap.add_argument('--skip', type=int, default=0); ap.add_argument('--n', type=int, default=1); ap.add_argument('--noise', type=float, default=0.005); ap.add_argument('--out', required=True); ap.add_argument('--T', type=int, default=16)
a = ap.parse_args(); T = a.T; R = []; t0 = time.time()
for name, img in load(12)[a.skip:a.skip + a.n]:
    torch.manual_seed(0); Y, d, fa = make_obs(img, name, a.noise); P = sample_points(d, fa, KS).to(dev) / HR
    r = dict(img=name, noise=a.noise, T=T, E=band_energy(img), arms={}); imgd = img.to(dev); nv = max(a.noise ** 2, 1e-5); var = float(Y.var()) * 1.0
    specs = {f'gp_gauss_{s}': gauss_P(s, var) for s in (3, 10, 30, 60, 128)}
    specs.update({f'gp_power_{al}': power_P(al, var) for al in (2.0, 2.5, 3.0)}); specs['gp_oracle'] = per_psd(imgd)
    SP = {'truth': spec(imgd)}
    for k, Ps in specs.items():
        xh = gp_map(Y, P, Ps, nv)
        r['arms'][k] = dict(psnr=psnr(xh, imgd), err=band_energy(xh - imgd), Pb=band_P(Ps)); SP[k] = spec(xh - imgd)
        print(name, a.noise, k, round(r['arms'][k]['psnr'], 2), f'{time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(a.out.replace('.json', f'_{name}_spec.npz'), **SP)
    R.append(r); json.dump(R, open(a.out, 'w')); print(name, a.noise, {k: round(v['psnr'], 2) for k, v in r['arms'].items()}, f'{time.time()-t0:.0f}s', flush=True)
print('done')
'''
open("/workspace/SE-INR/claude-20261005-d11/d11gp.py", "w").write("# D11: built from claude-20261004-e1/e1gp.py by mk_d11.py (cpu + band energies); prereg 预注册-D11-E1切换提前诊断-20261005.md\n" + src[:i] + main)
print("ok")
