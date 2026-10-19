"""Frozen-decoder gate intervention on the complete existing synthetic cohort.
No training, parameter update, GPU, or new image generation. Never resume a root.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,argparse,hashlib,resource,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled=False
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
sys.path.insert(0,str(Path(P['model_root'])/'snapshot/src'))
import models

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def maximum(x): return float(x.abs().max())
def coords():
    v=(np.arange(64)+.5)/64
    return torch.tensor(2*np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)-1,dtype=torch.float32)[None]

def geometry(net,q):
    assert net.local_ensemble and net.upinput
    areas=[]
    for vx in [-1,1]:
        for vy in [-1,1]:
            qq=q.clone()
            qq[:,:,0]+=vx/net.feat.shape[-2]+1e-6
            qq[:,:,1]+=vy/net.feat.shape[-1]+1e-6
            qq.clamp_(-1+1e-6,1-1e-6)
            qc=F.grid_sample(net.feat_coord,qq.flip(-1).unsqueeze(1),mode='nearest',align_corners=False)[:,:,0,:].permute(0,2,1)
            rel=q-qc
            rel[:,:,0]*=net.feat.shape[-2];rel[:,:,1]*=net.feat.shape[-1]
            areas.append(torch.abs(rel[:,:,0]*rel[:,:,1])+1e-9)
    total=torch.stack(areas).sum(0)
    areas=[areas[3],areas[2],areas[1],areas[0]]
    weights=torch.stack([(a/total)[0,:,None] for a in areas])
    skip=F.grid_sample(net.inp,q.flip(-1).unsqueeze(1),mode='bilinear',padding_mode='border',align_corners=False)[:,:,0,:].permute(0,2,1)[0]
    return weights,skip

def decoder_layers(net):
    ls=list(net.imnet.layers)
    assert len(ls)==7 and all(isinstance(ls[i],torch.nn.Linear) for i in [0,2,4,6])
    assert all(isinstance(ls[i],torch.nn.ReLU) for i in [1,3,5])
    return [(ls[i].weight.detach().double().clone(),ls[i].bias.detach().double().clone()) for i in [0,2,4,6]]

def decode(z,layers,delta=None):
    """Exact affine operations and ReLU directional derivative, zero inactive."""
    h=z;v=delta;masks=[];near=[]
    for i,(w,b) in enumerate(layers):
        h=F.linear(h,w,b)
        if v is not None:v=F.linear(v,w,None)
        if i<3:
            mask=h>0;masks.append(mask)
            near.append(float((h.abs()<=P['numeric']['near_preactivation']).double().mean()))
            h=h.clamp_min(0)
            if v is not None:v=v*mask
    return h,v,masks,near

def frozen_decode(z,layers,masks):
    h=z
    for i,(w,b) in enumerate(layers):
        h=F.linear(h,w,b)
        if i<3:h=h*masks[i]
    return h

def probe_checks(z0,zn,layers):
    """Independent autograd JVP and fixed-gate affine finite step, first8queries."""
    z0=z0[:8].clone();zn=zn[:8].clone();dz=zn-z0;checks={}
    for name,a in [('ref',z0),('mid',(z0+zn)/2),('native',zn)]:
        out,v,masks,_=decode(a,layers,dz)
        _,jvp=torch.autograd.functional.jvp(lambda x:decode(x,layers)[0],a,dz,create_graph=False,strict=False)
        fd=frozen_decode(a+dz,layers,masks)-frozen_decode(a,layers,masks)
        _,zero,_,_=decode(a,layers,torch.zeros_like(dz))
        err=dict(jvp=maximum(v-jvp),frozen_affine=maximum(v-fd),zero=maximum(zero))
        assert max(err.values())<=P['numeric']['double_audit_atol'],err
        checks[name]=err
    return checks

def infer(source,root,guard,preflight):
    q=coords();journal=[];obs=sorted((source/'observations').glob('*.npy'))
    for name in P['models']:
        guard()
        ck=torch.load(Path(P['model_root'])/'snapshot/checkpoints'/name/'epoch-last.pth',map_location='cpu',weights_only=True)
        assert ck['epoch']==1000
        net=models.make(ck['model'],load_sd=True).cpu().eval()
        for p in net.parameters():p.requires_grad_(False)
        layers=decoder_layers(net)
        np.savez_compressed(root/'probes'/(name+'__weights.npz'),**{f'{typ}{i}':a.numpy() for i,wb in enumerate(layers) for typ,a in zip(['w','b'],wb)})
        for j,f in enumerate(obs):
            guard();tic=time.monotonic();record=dict(model=name,observation=f.stem)
            try:
                y=np.load(f);inp=torch.from_numpy(np.tile(y[None],(3,1,1)))[None]
                captured=[]
                def hook(module,args,out):captured.append(args[0].detach().clone())
                handle=net.imnet.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        net.gen_feat(inp*2-1)
                        p0=(net.query_rgb(q,torch.zeros_like(q))[0]*.5+.5).clone()
                        assert len(captured)==4
                        z0=captured[:];captured.clear()
                        pn=(net.query_rgb(q,torch.full_like(q,2/64))[0]*.5+.5).clone()
                        assert len(captured)==4
                        zn=captured[:];captured.clear()
                        weights,skip=geometry(net,q)
                finally:handle.remove()
                old=np.load(source/'predictions'/(f.stem+'__'+name+'.npz'))
                endpoint=max(np.max(abs(old['point']-p0.numpy().reshape(64,64,3))),np.max(abs(old['native']-pn.numpy().reshape(64,64,3))))
                assert endpoint<=P['numeric']['endpoint_atol'],endpoint
                outputs={k:torch.zeros_like(p0,dtype=torch.float64) for k in ['point64','native64','d_ref','d_mid','d_native']}
                stats=[];probes={};tests=[]
                for branch,(a,b,ww) in enumerate(zip(z0,zn,weights)):
                    guard();a=a.double();b=b.double();dz=b-a
                    da,vr,ma,nr=decode(a,layers,dz)
                    db,vn,mn,nn=decode(b,layers,dz)
                    _,vm,mm,nm=decode((a+b)/2,layers,dz)
                    outputs['point64']+=ww.double()*da*.5
                    outputs['native64']+=ww.double()*db*.5
                    for key,v in [('d_ref',vr),('d_mid',vm),('d_native',vn)]:outputs[key]+=ww.double()*v*.5
                    stats.append(dict(branch=branch,mask_switch_ref_native=[float((x!=y).double().mean()) for x,y in zip(ma,mn)],mask_switch_ref_mid=[float((x!=y).double().mean()) for x,y in zip(ma,mm)],mask_switch_mid_native=[float((x!=y).double().mean()) for x,y in zip(mm,mn)],near_zero=dict(ref=nr,mid=nm,native=nn)))
                    probes[f'z0_{branch}']=a[:8].numpy();probes[f'zn_{branch}']=b[:8].numpy();probes[f'weight_{branch}']=ww[:8].numpy()
                    # All first8 probes retained. JVP all preflight, formal firstobs only/model.
                    if preflight or j==0:tests.append(dict(branch=branch,checks=probe_checks(a,b,layers)))
                for key in ['point64','native64']:outputs[key]+=skip.double()*.5+.5
                rebuild=max(maximum(outputs['point64']-p0.double()),maximum(outputs['native64']-pn.double()))
                assert rebuild<=P['numeric']['endpoint_atol'],rebuild
                outputs['d_full']=outputs['native64']-outputs['point64']
                outputs['residual']=outputs['d_full']-outputs['d_ref']
                arrays={k:v.numpy().reshape(64,64,3) for k,v in outputs.items()}
                arrays.update(point=p0.numpy().reshape(64,64,3),native=pn.numpy().reshape(64,64,3))
                assert all(np.isfinite(x).all() for x in arrays.values())
                np.savez_compressed(root/'predictions'/(f.stem+'__'+name+'.npz'),**arrays)
                np.savez_compressed(root/'probes'/(f.stem+'__'+name+'.npz'),**probes)
                record.update(status='complete',endpoint_float32_maxdiff=float(endpoint),rebuild_double_maxdiff=rebuild,mask_statistics=stats,probe_checks=tests)
            except BaseException:
                record.update(status='failed',traceback=traceback.format_exc());raise
            finally:
                record['seconds']=time.monotonic()-tic;journal.append(record);write(root/'inference_journal.json',journal)
            if j%8==0:print('INFER',name,j+1,len(obs),round(record['seconds'],3),flush=True)
        del net,ck
    manifest={str(f.relative_to(root)):sha(f) for folder in ['predictions','probes'] for f in sorted((root/folder).glob('*'))}
    write(root/'prediction_cohort.json',manifest)
    write(root/'prediction_freeze.json',dict(epoch=time.time(),manifest_sha256=sha(root/'prediction_cohort.json'),prediction_files=len(list((root/'predictions').glob('*.npz')))))

def planes(z,lo,hi):
    z=np.asarray(z,dtype=np.float64)[lo:hi,lo:hi]
    if z.ndim==2:z=np.repeat(z[:,:,None],3,2)
    f=np.fft.fftfreq(len(z));r=np.maximum(abs(f[:,None]),abs(f[None,:]));ff=np.fft.fft2(z,axes=(0,1));ans=[z]
    for low,high in [(0,.125),(.125,.25),(.25,.5001)]:
        mask=(r>=low)&(r<high)&(r>0)
        ans.append(np.fft.ifft2(ff*mask[:,:,None],axes=(0,1)).real)
    return ans
def sign(x,t):return 'positive' if x>t else 'negative' if x< -t else 'boundary'

def score(source,root,cases,guard):
    assert (root/'prediction_freeze.json').exists()
    rows=[]
    for c in cases:
        guard();truth=np.load(source/'truth'/(c['family']+'.npz'))['target']
        for model in P['models']:
            z=np.load(root/'predictions'/(c['observation']+'__'+model+'.npz'))
            for roi,(lo,hi) in P['evaluation']['ROIs'].items():
                pp={k:planes(z[k],lo,hi) for k in ['point','native','point64','d_ref','d_mid','d_native','d_full']}
                for band,t in enumerate(planes(truth,lo,hi)):
                    e=t-pp['point64'][band];dref=pp['d_ref'][band];r=pp['d_full'][band]-dref
                    row=dict(**c,model=model,roi=roi,band=band,reference_mse=float(np.mean(e*e)))
                    for mode in ['ref','mid','native','full']:
                        d=pp['d_'+mode][band];a=float(np.mean(e*d));b=float(np.mean(d*d));g=2*a-b
                        row.update({f'a_{mode}':a,f'b_{mode}':b,f'g_{mode}':g,f'direction_{mode}':sign(a,1e-10),f'gain_sign_{mode}':sign(g,1e-8)})
                    e32=t-pp['point'][band];d32=pp['native'][band]-pp['point'][band]
                    a32=float(np.mean(e32*d32));g32=float(2*a32-np.mean(d32*d32))
                    row.update(old_float32_alignment=a32,old_float32_gain=g32,numerical_sensitive_direction=sign(a32,1e-10)!=row['direction_full'],numerical_sensitive_gain=sign(g32,1e-8)!=row['gain_sign_full'])
                    row.update(residual_alignment=float(np.mean(e*r)),residual_energy=float(np.mean(r*r)),residual_gain_cross=float(2*np.mean((e-dref)*r)),residual_gain_penalty=float(np.mean(r*r)))
                    row['gain_decomposition_residual']=row['g_full']-row['g_ref']-row['residual_gain_cross']+row['residual_gain_penalty']
                    assert abs(row['gain_decomposition_residual'])<1e-12
                    assert abs(row['a_full']-row['a_ref']-row['residual_alignment'])<1e-12
                    rows.append(row)
    write(root/'rows.json',rows)
    groups={}
    for model in P['models']:
        for roi in P['evaluation']['ROIs']:
            for band in range(4):
                rr=[r for r in rows if (r['model'],r['roi'],r['band'])==(model,roi,band)]
                dn=[r for r in rr if r['direction_full']=='negative' and not r['numerical_sensitive_direction']]
                groups[f'{model}/{roi}/b{band}']=dict(n=len(rr),full_direction_negative=len(dn),ref_direction_among_full_negative={s:sum(r['direction_ref']==s for r in dn) for s in ['positive','negative','boundary']},all_anchor_negative_among_full_negative=sum(all(r['direction_'+m]=='negative' for m in ['ref','mid','native']) for r in dn),all_anchor_positive_among_full_negative=sum(all(r['direction_'+m]=='positive' for m in ['ref','mid','native']) for r in dn),direction_transition={a+'->'+b:sum(r['direction_ref']==a and r['direction_full']==b for r in rr) for a in ['positive','negative','boundary'] for b in ['positive','negative','boundary']},gain_transition={a+'->'+b:sum(r['gain_sign_ref']==a and r['gain_sign_full']==b for r in rr) for a in ['positive','negative','boundary'] for b in ['positive','negative','boundary']},mean_gain={m:float(np.mean([r['g_'+m] for r in rr])) for m in ['ref','mid','native','full']})
                groups[f'{model}/{roi}/b{band}'].update(numerical_sensitive_direction=sum(r['numerical_sensitive_direction'] for r in rr),numerical_sensitive_gain=sum(r['numerical_sensitive_gain'] for r in rr),transition_note='Unqualified transition matrices include sensitive rows; stable matrices below exclude corresponding sensitive rows.',direction_transition_stable={a+'->'+b:sum(r['direction_ref']==a and r['direction_full']==b and not r['numerical_sensitive_direction'] for r in rr) for a in ['positive','negative','boundary'] for b in ['positive','negative','boundary']},gain_transition_stable={a+'->'+b:sum(r['gain_sign_ref']==a and r['gain_sign_full']==b and not r['numerical_sensitive_gain'] for r in rr) for a in ['positive','negative','boundary'] for b in ['positive','negative','boundary']})
    pairs=[];lookup={(r['family'],r['model'],r['roi'],r['band'],r['eta']):r for r in rows}
    for r0 in rows:
        if r0['eta']!=0:continue
        r1=lookup[(r0['family'],r0['model'],r0['roi'],r0['band'],1)]
        item={k:r0[k] for k in ['family','template','phase','shift','model','roi','band']}
        item['numerical_sensitive_direction']=r0['numerical_sensitive_direction'] or r1['numerical_sensitive_direction']
        item['numerical_sensitive_gain']=r0['numerical_sensitive_gain'] or r1['numerical_sensitive_gain']
        item['old32_positive_to_negative']=r0['old_float32_alignment']>1e-10 and r1['old_float32_alignment']<-1e-10
        for m in ['ref','mid','native','full']:
            item[m+'_positive_to_negative']=r0['a_'+m]>1e-10 and r1['a_'+m]<-1e-10
            item['a0_'+m]=r0['a_'+m];item['a1_'+m]=r1['a_'+m];item['delta_a_'+m]=r1['a_'+m]-r0['a_'+m]
        item['delta_a_residual']=r1['residual_alignment']-r0['residual_alignment'];pairs.append(item)
    write(root/'eta_pairs.json',pairs)
    write(root/'numerical_sensitive.json',dict(direction=[r for r in rows if r['numerical_sensitive_direction']],gain=[r for r in rows if r['numerical_sensitive_gain']]))
    oldmid={model:{key:sum(r[key] for r in pairs if r['model']==model and r['roi']=='center' and r['band']==2 and r['template'].startswith('mid_')) for key in ['old32_positive_to_negative','full_positive_to_negative','numerical_sensitive_direction']} for model in P['models']}
    write(root/'summary.json',dict(groups=groups,midband_center_reproduction=oldmid,rows=len(rows),conditions=len(cases),scope=P['scope'],training=0,GPU=0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--launch',required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    root=Path(args.out);root.mkdir()
    for d in ['predictions','probes']:(root/d).mkdir()
    launch=json.loads(Path(args.launch).read_text());write(root/'launch.json',launch)
    start=time.monotonic();budget=P['budget']['preflight_seconds'] if args.preflight else P['budget']['formal_seconds']
    assert launch['deadline_epoch']-launch['start_epoch']<=budget+1e-6
    status='failed';phase={}
    def guard():
        if time.time()>launch['deadline_epoch'] or time.monotonic()-start>budget:raise TimeoutError('fixed absolute/cumulative budget exhausted; no resume')
    try:
        guard();tic=time.monotonic()
        source=Path(P['preflight_root'] if args.preflight else P['formal_root'])
        frozen=P['source_manifests']['preflight' if args.preflight else 'formal']
        for filename,h in frozen.items():assert sha(source/filename)==h,('frozen manifest drift',filename)
        inherited=json.loads((source/'inherited_manifest.json').read_text())
        for p,h in inherited.items():guard();assert sha(p)==h,('inherited drift',p)
        for manifestname in ['input_manifest.json','prediction_cohort.json']:
            for p,h in json.loads((source/manifestname).read_text()).items():guard();assert sha(source/p)==h,('old cohort drift',p)
        cases=json.loads((source/'cases.json').read_text());write(root/'cases.json',cases)
        count=len(list((source/'observations').glob('*.npy')))
        assert (len(cases),count)==((3,3) if args.preflight else (240,148))
        write(root/'input_freeze.json',dict(epoch=time.time(),source_root=str(source),source_sha256=sha(HERE/'run.py'),protocol_sha256=sha(HERE/'protocol.json'),input_manifest_sha256=sha(source/'input_manifest.json'),old_prediction_cohort_sha256=sha(source/'prediction_cohort.json'),inherited_manifest_sha256=sha(source/'inherited_manifest.json'),conditions=len(cases),observations=count,preflight=args.preflight))
        phase['prepare']=time.monotonic()-tic;tic=time.monotonic()
        infer(source,root,guard,args.preflight);phase['inference']=time.monotonic()-tic;tic=time.monotonic()
        score(source,root,cases,guard);phase['score']=time.monotonic()-tic;guard()
        status='complete';write(root/'COMPLETE.json',dict(epoch=time.time(),rows_sha256=sha(root/'rows.json'),summary_sha256=sha(root/'summary.json')))
    except BaseException:
        (root/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF)
        write(root/'resources.json',dict(status=status,wall_seconds=time.time()-launch['start_epoch'],script_seconds=time.monotonic()-start,cpu_seconds=u.ru_utime+u.ru_stime,peak_rss_kib=u.ru_maxrss,phase_seconds=phase,CPU_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),GPU=0,training=0,end_epoch=time.time()))
if __name__=='__main__':main()
