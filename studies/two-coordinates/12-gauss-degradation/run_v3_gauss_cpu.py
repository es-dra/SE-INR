"""Within-scale cell-mismatch sweep. Inference only. Writes to --out only."""
import argparse, hashlib, json, os, sys, time, glob, traceback
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = Path(__file__).resolve().parent
P = json.loads((HERE/'protocol.json').read_text()); P['scales']=[2,3,4,6]
ap0 = argparse.ArgumentParser(add_help=False); ap0.add_argument('--src', required=True); ap0.add_argument('--ckpt', required=True); ap0.add_argument('--name', required=True); ap0.add_argument('--needs-scale', action='store_true'); A0, _ = ap0.parse_known_args()
sys.path.insert(0, A0.src); import models; from datasets.gauss_wrapper import gauss_down
CK = '/workspace/SE-INR/SC-ASISR/artifacts/checkpoints/seed{}/{}/epoch-last.pth'
DATA = {'DIV2K-fresh': '/workspace/SE-INR/Data/DIV2K_valid_HR'}
H = 240; SH = 4; N = H - 2*SH
def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()
def images(ds, limit):
    fs = [p for p in sorted(glob.glob(DATA[ds]+'/*.png')) if 869 <= int(Path(p).stem) <= 900]
    return fs[:limit] if limit else fs
def crop(p):
    im = Image.open(p).convert('RGB'); w, h = im.size; assert min(w, h) >= H, p
    l, t = (w-H)//2, (h-H)//2
    return np.asarray(im.crop((l, t, l+H, t+H)), dtype='float32')/255
f = np.fft.fftfreq(N); R = np.maximum(abs(f[:, None]), abs(f[None, :]))
def spec(x): return np.fft.fft2(x.astype('float64')[SH:H-SH, SH:H-SH], axes=(0, 1))
def masks(s):
    u = 2*s*R; edges = np.arange(0, s+0.25, 0.25); edges[-1] = s + 1e-6
    return [(float(lo), float(hi), (u >= lo) & (u < hi) & (R > 0)) for lo, hi in zip(edges[:-1], edges[1:])]
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--save-pred', default='none'); ap.add_argument('--src'); ap.add_argument('--ckpt'); ap.add_argument('--name'); ap.add_argument('--needs-scale', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=False); (out/'pred').mkdir()
    dev = torch.device('cpu'); torch.set_num_threads(64); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    man = {'source': {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*.py'))+[HERE/'protocol.json'] if '__pycache__' not in p.parts}, 'checkpoints': {}, 'images': {}}
    t0 = time.time(); rows = []; ms = P['cell_multipliers']; nrm = N*N*N*N*3
    v = (np.arange(H)+.5)/H; q = torch.from_numpy((2*np.stack(np.meshgrid(v, v, indexing='ij'), -1).reshape(-1, 2)-1).astype('float32'))[None].to(dev)
    data = {ds: [(Path(p).stem, crop(p)) for p in images(ds, a.limit)] for ds in DATA}
    for ds in DATA:
        for p in images(ds, a.limit): man['images'][p] = sha(p)
    for name, seed in [(A0.name, 0)]:
        if True:
            ck = A0.ckpt; man['checkpoints'][ck] = sha(ck)
            c = torch.load(ck, map_location='cpu', weights_only=False)
            m = models.make(c['model'], load_sd=True).to(dev).eval()
            for ds, ims in data.items():
                for stem, x in ims:
                    T = spec(x)
                    for s in P['scales']:
                        y = gauss_down(torch.from_numpy(x.transpose(2, 0, 1)), (H//s, H//s))[None]
                        m.gen_feat((y*2-1).to(dev))
                        preds = {}
                        for mu in ms:
                            cell = torch.full_like(q, mu*2/H)
                            z = (m.query_rgb(q, torch.full((1,), float(s), device=dev), cell) if A0.needs_scale else m.query_rgb(q, cell))[0].float().cpu().numpy()*.5+.5
                            assert np.isfinite(z).all(); preds[mu] = z.reshape(H, H, 3)
                        if ds == a.save_pred:
                            np.savez_compressed(out/'pred'/f'{ds}_{stem}_x{s}_{name}_s{seed}.npz', **{str(k): v.astype('float16') for k, v in preds.items()})
                        S = {mu: spec(z) for mu, z in preds.items()}; P1 = S[1.0]
                        for lo, hi, mk in masks(s):
                            rec = dict(ds=ds, img=stem, model=name, seed=seed, s=s, lo=lo, hi=hi, tgt=float((abs(T[mk])**2).sum()/nrm), err={}, a={}, b={})
                            E1 = T - P1
                            for mu in ms:
                                D = S[mu] - P1
                                rec['err'][str(mu)] = float((abs(T[mk]-S[mu][mk])**2).sum()/nrm)
                                rec['a'][str(mu)] = float((E1[mk].conj()*D[mk]).real.sum()/nrm)
                                rec['b'][str(mu)] = float((abs(D[mk])**2).sum()/nrm)
                            rows.append(rec)
                    print(name, seed, ds, stem, round(time.time()-t0, 1), flush=True)
            del m; torch.cuda.empty_cache()
    (out/'manifest.json').write_text(json.dumps(man, indent=1))
    (out/'rows.json').write_text(json.dumps(rows))
    (out/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), seconds=time.time()-t0, training_runs=0)))
if __name__ == '__main__':
    try: main()
    except Exception: print(traceback.format_exc()); raise
