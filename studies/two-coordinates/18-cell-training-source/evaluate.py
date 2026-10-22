"""Fixed epoch-200, complete benchmark Y-PSNR and DIV2K R(u). No model fitting."""
import argparse, gzip, hashlib, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--arm',choices=['native','nocell','constant','shuffle'],required=True);p.add_argument('--seed',type=int,choices=[1,2],required=True);a=p.parse_args()
r=Path(a.run_root).resolve();sys.path[:0]=[str(r/'source'),str(r)];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import models, utils
from datasets.wrappers import SRImplicitDownsampled
import cell_source
P=json.loads((r/'protocol.json').read_text());expected_images={k:v for items in json.loads((r/'data_manifest.json').read_text()).values() for k,v in items.items()};out=r/'evaluation'/f'{a.arm}_s{a.seed}'
if out.exists(): raise FileExistsError(out)
out.mkdir(parents=True);t0=time.time()
ck=Path(P['baseline_checkpoints'][f'{a.arm}_s{a.seed}']['path']) if a.arm in ('native','nocell') else r/'save'/f'{a.arm}_s{a.seed}'/'epoch-last.pth'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if a.arm in ('native','nocell'): assert sha(ck)==P['baseline_checkpoints'][f'{a.arm}_s{a.seed}']['sha256']
c=torch.load(ck,map_location='cpu',weights_only=False);assert c['epoch']==200
m=models.make(c['model'],load_sd=True).cuda().eval()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
MODES=['native','mean']+[f'marginal_{i}' for i in range(4)] if a.arm=='shuffle' else ['native']
rows=0;image_hash={}

def marginal(ds,img,j):
    # Same image's assigned value is held across scales; independent of target scale.
    k=f'cell-source-20261023:{ds}:{img}:{j}'
    v=int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],'big')/2**64
    return 96.0/round(48*(1+3*v))

@torch.no_grad()
def predict(inp,coord,cell,ds,img,mode):
    if a.arm=='shuffle':
        m.eval_cell_mode='marginal' if mode.startswith('marginal_') else mode
        if mode.startswith('marginal_'):m.eval_cell_value=marginal(ds,img,int(mode.rsplit('_',1)[1]))
    m.gen_feat((inp*2-1).cuda());z=[]
    coord=coord.cuda();cell=cell.cuda()
    for st in range(0,coord.shape[1],50000):
        z.append(m.query_rgb(coord[:,st:st+50000],cell[:,st:st+50000]).float().cpu())
    return torch.cat(z,dim=1)*.5+.5

with gzip.open(out/'rows.jsonl.gz','wt') as f:
    for ds in ['Set5','Set14','BSD100','Urban100']:
        for path in sorted((Path(P['data_root'])/ds/'HR').glob('*.png')):
            image_hash[str(path)]=sha(path);assert image_hash[str(path)]==expected_images[str(path)];x=torch.from_numpy(np.asarray(Image.open(path).convert('RGB'),dtype='float32').copy()).permute(2,0,1)/255
            for s in P['scales']:
                b=SRImplicitDownsampled([x],scale_min=s,scale_max=s)[0];inp=b['inp'][None];H,W=inp.shape[-2]*s,inp.shape[-1]*s
                gt=b['gt'].view(H,W,3).permute(2,0,1)[None]
                for mode in MODES:
                    z=predict(inp,b['coord'][None],b['cell'][None],ds,path.stem,mode).view(1,H,W,3).permute(0,3,1,2).clamp(0,1)
                    v=float(utils.calc_psnr(z,gt,dataset='benchmark',scale=s).item())
                    row={'kind':'benchmark','dataset':ds,'image':path.stem,'scale':s,'arm':a.arm,'seed':a.seed,'eval_mode':mode,'y_psnr':v}
                    assert np.isfinite(v);f.write(json.dumps(row)+'\n');rows+=1
            print('benchmark',a.arm,a.seed,ds,path.stem,round(time.time()-t0,2),flush=True)
    H=240;shave=4;N=H-2*shave;freq=np.fft.fftfreq(N);rad=np.maximum(abs(freq[:,None]),abs(freq[None,:]));norm=3*N**4
    coord=utils.make_coord((H,H))[None];cell=torch.ones_like(coord)*2/H
    for path in sorted((Path(P['data_root'])/'DIV2K_valid_HR').glob('*.png')):
        if not 869<=int(path.stem)<=900: continue
        image_hash[str(path)]=sha(path);assert image_hash[str(path)]==expected_images[str(path)];im=Image.open(path).convert('RGB');w,h=im.size;l,t=(w-H)//2,(h-H)//2
        x=np.asarray(im.crop((l,t,l+H,t+H)),dtype='float32')/255
        T=np.fft.fft2(x[shave:-shave,shave:-shave].astype('float64'),axes=(0,1))
        for s in P['scales']:
            inp=F.interpolate(torch.from_numpy(x.transpose(2,0,1))[None],size=(H//s,H//s),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            edges=np.arange(0,s+.25,.25);edges[-1]=s+1e-6;u=2*s*rad
            for mode in MODES:
                z=predict(inp,coord,cell,'DIV2K-fresh',path.stem,mode).numpy().reshape(H,H,3)
                Z=np.fft.fft2(z[shave:-shave,shave:-shave].astype('float64'),axes=(0,1));error=abs(T-Z)**2;energy=abs(T)**2
                for lo,hi in zip(edges[:-1],edges[1:]):
                    mask=(u>=lo)&(u<hi)&(rad>0)
                    row={'kind':'recovery','dataset':'DIV2K-fresh','image':path.stem,'scale':s,'arm':a.arm,'seed':a.seed,'eval_mode':mode,'lo':float(lo),'hi':float(hi),'target':float(energy[mask].sum()/norm),'error':float(error[mask].sum()/norm)}
                    assert np.isfinite(row['error']);f.write(json.dumps(row)+'\n');rows+=1
        print('recovery',a.arm,a.seed,path.stem,round(time.time()-t0,2),flush=True)
(out/'COMPLETE.json').write_text(json.dumps({'rows':rows,'rows_sha256':sha(out/'rows.jsonl.gz'),'frozen_sha256':sha(r/'FROZEN.sha256'),'seconds':time.time()-t0,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'image_sha256':image_hash,'modes':MODES,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},indent=2)+'\n')
