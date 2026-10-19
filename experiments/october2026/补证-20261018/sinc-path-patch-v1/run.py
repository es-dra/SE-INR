"""Frozen SC sinc-envelope intervention. Existing checkpoint/inputs; no training.
New roots only. Predictions freeze before scoring; retrospective cohort explicitly.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,argparse,hashlib,resource,traceback
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
sys.path.insert(0,str(Path(P['model_root'])/'snapshot/src'))
import models

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def maxabs(x):return float(x.abs().max())
def rms(x):return float(torch.mean(x*x).sqrt())
def coords():
    v=(np.arange(64)+.5)/64
    return torch.tensor(2*np.stack(np.meshgrid(v,v,indexing='ij'),-1).reshape(-1,2)-1,dtype=torch.float32)[None]
def layers_of(net):
    ls=list(net.imnet.layers)
    assert len(ls)==7 and all(isinstance(ls[i],torch.nn.Linear) for i in [0,2,4,6])
    assert all(isinstance(ls[i],torch.nn.ReLU) for i in [1,3,5])
    return [(ls[i].weight.detach().double().clone(),ls[i].bias.detach().double().clone()) for i in [0,2,4,6]]
def decode(z,layers):
    for i,(w,b) in enumerate(layers):
        z=F.linear(z,w,b)
        if i<3:z=z.clamp_min(0)
    return z
def geometry_and_envelope(net,q):
    assert net.local_ensemble and net.upinput and net.use_sinc_response
    areas=[];envelopes=[];omega_probes=[]
    for vx in [-1,1]:
        for vy in [-1,1]:
            qq=q.clone();qq[:,:,0]+=vx/net.feat.shape[-2]+1e-6;qq[:,:,1]+=vy/net.feat.shape[-1]+1e-6
            qq.clamp_(-1+1e-6,1-1e-6)
            def fetch(z):return F.grid_sample(z,qq.flip(-1).unsqueeze(1),mode='nearest',align_corners=False)[:,:,0,:].permute(0,2,1)
            qc=fetch(net.feat_coord);omega=fetch(net.omega_map).reshape(1,4096,net.num_freqs,2)
            omega_probes.append(omega[0,P['numeric']['probe_query_indices']].clone())
            rel=q-qc;rel[:,:,0]*=net.feat.shape[-2];rel[:,:,1]*=net.feat.shape[-1]
            areas.append(torch.abs(rel[:,:,0]*rel[:,:,1])+1e-9)
            cell=torch.full_like(q,2/64);cell[:,:,0]*=net.feat.shape[-2];cell[:,:,1]*=net.feat.shape[-1]
            w=torch.sinc(omega[:,:,:,0]*cell[:,:,0:1]/2)*torch.sinc(omega[:,:,:,1]*cell[:,:,1:2]/2)
            envelopes.append(torch.cat([w,w],-1)[0])
    total=torch.stack(areas).sum(0)
    weights=torch.stack([(areas[j]/total)[0,:,None] for j in [3,2,1,0]])
    skip=F.grid_sample(net.inp,q.flip(-1).unsqueeze(1),mode='bilinear',padding_mode='border',align_corners=False)[:,:,0,:].permute(0,2,1)[0]
    return torch.stack(envelopes),weights,skip,torch.stack(omega_probes)

def infer(source,root,cohort,guard):
    ck=torch.load(Path(P['model_root'])/'snapshot/checkpoints/sc-inr/epoch-last.pth',map_location='cpu',weights_only=True)
    assert ck['epoch']==P['checkpoint_epoch']
    net=models.make(ck['model'],load_sd=True).cpu().eval()
    for p in net.parameters():p.requires_grad_(False)
    layers=layers_of(net);q=coords();cache=OrderedDict();observation_journal=[];journal=[]
    np.savez_compressed(root/'probes/weights.npz',**{f'{t}{i}':a.numpy() for i,wb in enumerate(layers) for t,a in zip(['w','b'],wb)})
    permutation=np.random.default_rng(20261018).permutation(128);paired=np.concatenate([permutation,permutation+128]);indices=P['numeric']['probe_query_indices']
    def observation(name):
        guard()
        if name in cache:cache.move_to_end(name);return cache[name]
        tic=time.monotonic();y=np.load(source/'observations'/(name+'.npy'));inp=torch.from_numpy(np.tile(y[None],(3,1,1)))[None]
        captured=[]
        def hook(module,args,out):captured.append(args[0].detach().clone())
        handle=net.imnet.register_forward_hook(hook)
        try:
            with torch.no_grad():
                net.gen_feat(inp*2-1)
                p0=(net.query_rgb(q,torch.zeros_like(q))[0]*.5+.5).clone();assert len(captured)==4
                U=torch.stack(captured);captured.clear()
                pn=(net.query_rgb(q,torch.full_like(q,2/64))[0]*.5+.5).clone();assert len(captured)==4
                native_features=torch.stack(captured);captured.clear()
                W,weights,skip,omega=geometry_and_envelope(net,q)
        finally:handle.remove()
        feature_error=maxabs(native_features-U*W)
        assert feature_error<=P['numeric']['reconstructed_feature_atol'],('feature parity',name,feature_error)
        old=np.load(source/'predictions'/(name+'__sc-inr.npz'))
        endpoint=max(float(np.max(abs(old['point']-p0.numpy().reshape(64,64,3)))),float(np.max(abs(old['native']-pn.numpy().reshape(64,64,3)))))
        assert endpoint<=P['numeric']['endpoint_atol'],('endpoint',name,endpoint)
        result=dict(U=U,W=W,weights=weights,skip=skip,point=p0,native=pn,omega=omega)
        observation_journal.append(dict(observation=name,endpoint_float32_maxdiff=endpoint,feature_product_maxdiff=feature_error,seconds=time.monotonic()-tic))
        write(root/'observation_journal.json',observation_journal)
        cache[name]=result
        while len(cache)>5:cache.popitem(last=False)
        return result
    for i,c in enumerate(cohort):
        guard();tic=time.monotonic();record=dict(family=c['family'])
        try:
            a=observation(c['eta0']);b=observation(c['eta1'])
            U0=a['U'].double();U1=b['U'].double();W0=a['W'].double();W1=b['W'].double()
            assert torch.equal(a['weights'],b['weights'])
            weights=a['weights'].double();s0=a['skip'].double();s1=b['skip'].double()
            target=rms(U1*(W0-1));u_rms=rms(U1);assert u_rms>0
            t=target/u_rms;perm_w=W0[:,:,paired];den=rms(U1*(perm_w-1));assert den>0 or target==0
            gamma=target/den if den else 0.;uniform=torch.full_like(W0,1-t);permuted=1-gamma*(1-perm_w)
            assert abs(rms(U1*(uniform-1))-target)<=1e-12
            assert abs(rms(U1*(permuted-1))-target)<=1e-12
            definitions={'point0':(U0,None,s0),'native0':(U0,W0,s0),'point1':(U1,None,s1),'native1':(U1,W1,s1),'swap10':(U1,W0,s1),'swap01':(U0,W1,s0),'uniform1':(U1,uniform,s1),'permuted1':(U1,permuted,s1)}
            outputs={}
            for name,(u,w,skip) in definitions.items():
                guard();out=skip*.5+.5
                for branch in range(4):out=out+weights[branch]*decode(u[branch] if w is None else u[branch]*w[branch],layers)*.5
                outputs[name]=out.numpy().reshape(64,64,3)
            parity=max(float(np.max(abs(outputs[f'{mode}{eta}']-v[mode].numpy().reshape(64,64,3)))) for eta,v in [(0,a),(1,b)] for mode in ['point','native'])
            assert parity<=P['numeric']['endpoint_atol'],('double endpoint',c['family'],parity)
            sham=max(float(np.max(abs(outputs['swap10']-outputs['native1']))),float(np.max(abs(outputs['swap01']-outputs['native0'])))) if c['template']=='control_low' else None
            if sham is not None:assert sham<=P['numeric']['double_probe_atol']
            outputs.update({f'old_{mode}{eta}':v[mode].numpy().reshape(64,64,3) for eta,v in [(0,a),(1,b)] for mode in ['point','native']})
            assert all(np.isfinite(v).all() for v in outputs.values())
            np.savez_compressed(root/'predictions'/(c['family']+'.npz'),**outputs)
            probes={k:x[:,indices].numpy() for k,x in [('U0',U0),('U1',U1),('W0',W0),('W1',W1),('weights',weights)]}
            probes.update(omega0=a['omega'].numpy(),omega1=b['omega'].numpy(),rel_cell=np.array([.5,.5],dtype=np.float32),skip0=s0[indices].numpy(),skip1=s1[indices].numpy(),indices=np.array(indices),permutation=permutation,uniform_t=np.array(t),permuted_gamma=np.array(gamma),target_feature_rms=np.array(target),U1_rms=np.array(u_rms),permuted_unscaled_rms=np.array(den))
            np.savez_compressed(root/'probes'/(c['family']+'.npz'),**probes)
            energy=U1*U1;energy_sum=float(energy.sum());assert energy_sum>0
            record.update(status='complete',rebuild_double_maxdiff=parity,sham_maxdiff=sham,uniform_t=t,permuted_gamma=gamma,target_feature_rms=target,uniform_feature_rms=rms(U1*(uniform-1)),permuted_feature_rms=rms(U1*(permuted-1)),U0_rms=rms(U0),U1_rms=u_rms,U_change_rms=rms(U1-U0),W_change_rms=rms(W1-W0),weighted_attenuation0=float((energy*(1-W0)).sum()/energy_sum),weighted_attenuation1=float((energy*(1-W1)).sum()/energy_sum),weighted_attenuation_change=float((energy*(W0-W1)).sum()/energy_sum),weights={name:dict(min=float(w.min()),max=float(w.max()),out_of_unit_count=int(((w<0)|(w>1)).sum()),count=w.numel()) for name,w in [('W0',W0),('W1',W1),('uniform',uniform),('permuted',permuted)]})
        except BaseException:record.update(status='failed',traceback=traceback.format_exc());raise
        finally:
            record['seconds']=time.monotonic()-tic;journal.append(record);write(root/'inference_journal.json',journal)
        if i%4==0:print('INFER',i+1,len(cohort),round(record['seconds'],3),flush=True)
    manifest={str(f.relative_to(root)):sha(f) for d in ['predictions','probes'] for f in sorted((root/d).glob('*'))}
    write(root/'prediction_cohort.json',manifest)
    write(root/'prediction_freeze.json',dict(epoch=time.time(),manifest_sha256=sha(root/'prediction_cohort.json'),families=len(cohort),prediction_files=len(cohort),probe_files=len(cohort)+1))

def planes(z,lo,hi):
    z=np.asarray(z,dtype=np.float64)[lo:hi,lo:hi]
    if z.ndim==2:z=np.repeat(z[:,:,None],3,2)
    f=np.fft.fftfreq(len(z));r=np.maximum(abs(f[:,None]),abs(f[None,:]));ff=np.fft.fft2(z,axes=(0,1));ans=[z]
    for low,high in [(0,.125),(.125,.25),(.25,.5001)]:
        mask=(r>=low)&(r<high)&(r>0);ans.append(np.fft.ifft2(ff*mask[:,:,None],axes=(0,1)).real)
    return ans

def score(source,root,cohort,guard):
    freeze=json.loads((root/'prediction_freeze.json').read_text());assert sha(root/'prediction_cohort.json')==freeze['manifest_sha256']
    rows=[];tg=P['scoring']['tol_gain'];ta=P['scoring']['tol_direction']
    for c in cohort:
        guard();truth=np.load(source/'truth'/(c['family']+'.npz'))['target'];z=np.load(root/'predictions'/(c['family']+'.npz'))
        for roi,(lo,hi) in {'center':(16,48),'full':(0,64)}.items():
            pp={k:planes(z[k],lo,hi) for k in z.files}
            for band,T in enumerate(planes(truth,lo,hi)):
                row=dict(**c,roi=roi,band=band)
                e0=T-pp['point0'][band];e1=T-pp['point1'][band]
                d0=pp['native0'][band]-pp['point0'][band];d1=pp['native1'][band]-pp['point1'][band]
                row['reference_mse0']=float(np.mean(e0*e0));row['reference_mse1']=float(np.mean(e1*e1))
                for name,eta in [('native0',0),('native1',1),('swap10',1),('swap01',0),('uniform1',1),('permuted1',1)]:
                    e=e0 if eta==0 else e1;d=pp[name][band]-pp[f'point{eta}'][band]
                    a=float(np.mean(e*d));b=float(np.mean(d*d));g=2*a-b;mse=float(np.mean((T-pp[name][band])**2))
                    assert abs(row[f'reference_mse{eta}']-mse-g)<P['numeric']['score_atol']
                    row.update({f'a_{name}':a,f'b_{name}':b,f'g_{name}':g,f'mse_{name}':mse})
                row.update(cross_e0_d0=float(np.mean(e0*d0)),cross_e0_d1=float(np.mean(e0*d1)),cross_e1_d0=float(np.mean(e1*d0)),cross_e1_d1=float(np.mean(e1*d1)),demand_swapped_mse=float(np.mean((e1-d0)**2)),demand_swapped_gain=float(2*np.mean(e1*d0)-np.mean(d0*d0)))
                row['alignment_change']=row['a_native1']-row['a_native0'];row['alignment_change_response']=float(np.mean(e0*(d1-d0)));row['alignment_change_demand']=float(np.mean((e1-e0)*d1))
                assert abs(row['alignment_change']-row['alignment_change_response']-row['alignment_change_demand'])<P['numeric']['score_atol']
                row['target_improvement']=row['mse_native1']-row['mse_swap10'];row['target_over_uniform']=row['mse_uniform1']-row['mse_swap10'];row['target_over_permuted']=row['mse_permuted1']-row['mse_swap10'];row['reverse_gain_decrease']=row['g_native0']-row['g_swap01']
                row['rescued']=row['g_native1']< -tg and row['g_swap10']>=-tg
                for eta in [0,1]:
                    e=T-pp[f'old_point{eta}'][band];d=pp[f'old_native{eta}'][band]-pp[f'old_point{eta}'][band]
                    row[f'old_a{eta}']=float(np.mean(e*d));row[f'old_g{eta}']=float(2*np.mean(e*d)-np.mean(d*d))
                    sg=lambda x,t:1 if x>t else -1 if x<-t else 0
                    row[f'sensitive_direction{eta}']=sg(row[f'old_a{eta}'],ta)!=sg(row[f'a_native{eta}'],ta)
                    row[f'sensitive_gain{eta}']=sg(row[f'old_g{eta}'],tg)!=sg(row[f'g_native{eta}'],tg)
                rows.append(row)
    write(root/'rows.json',rows);journal={r['family']:r for r in json.loads((root/'inference_journal.json').read_text())}
    groups={}
    selectors={'all32':lambda c:c['template'].startswith('mid_'),'focal16':lambda c:c['focal'],'nonflip16':lambda c:c['template'].startswith('mid_') and not c['focal'],'sham16':lambda c:c['template']=='control_low'}
    for name,sel in selectors.items():
        for roi in ['center','full']:
            for band in range(4):
                rr=[r for r in rows if sel(r) and r['roi']==roi and r['band']==band]
                if not rr:continue
                keys=['target_improvement','target_over_uniform','target_over_permuted','reverse_gain_decrease','alignment_change','alignment_change_response','alignment_change_demand','g_native0','g_native1','g_swap10','g_swap01','g_uniform1','g_permuted1']
                groups[f'{name}/{roi}/b{band}']=dict(n=len(rr),rescued=sum(r['rescued'] for r in rr),negative_native1=sum(r['g_native1']< -tg for r in rr),reverse_decreased=sum(r['reverse_gain_decrease']>tg for r in rr),median={k:float(np.median([r[k] for r in rr])) for k in keys},mean={k:float(np.mean([r[k] for r in rr])) for k in keys},sensitive_direction=sum(r['sensitive_direction0'] or r['sensitive_direction1'] for r in rr),sensitive_gain=sum(r['sensitive_gain0'] or r['sensitive_gain1'] for r in rr))
    focal=[r for r in rows if r['focal'] and r['roi']=='center' and r['band']==2]
    focal_rgb=[r for r in rows if r['focal'] and r['roi']=='center' and r['band']==0]
    gates={};premise=None
    if len(focal)==16:
        g=groups['focal16/center/b2'];median=g['median']
        gates=dict(rescue_majority=g['rescued']>8,targeted_improvement=median['target_improvement']>tg,beats_uniform=median['target_over_uniform']>tg,beats_permuted=median['target_over_permuted']>tg,reverse_median=median['reverse_gain_decrease']>tg,reverse_majority=g['reverse_decreased']>8,fullRGB_not_worse=float(np.median([r['target_improvement'] for r in focal_rgb]))>=-tg)
        premise=float(np.median([journal[r['family']]['weighted_attenuation_change'] for r in focal]))>ta
    write(root/'summary.json',dict(groups=groups,gates=gates,major_envelope_explanation_supported=all(gates.values()) if gates else None,stronger_attenuation_premise=premise,rows=len(rows),families=len(cohort),limitations=P['limitations'],scope=P['cohort_scope'],training=0,GPU=0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--launch',required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    root=Path(args.out);root.mkdir()
    for d in ['predictions','probes']:(root/d).mkdir()
    launch=json.loads(Path(args.launch).read_text());write(root/'launch.json',launch);start=time.monotonic()
    budget=P['budget']['preflight_seconds'] if args.preflight else P['budget']['formal_seconds']
    assert 0<launch['deadline_epoch']-launch['start_epoch']<=budget+1e-6
    phase={};status='failed'
    def guard():
        if time.time()>launch['deadline_epoch'] or time.monotonic()-start>budget:raise TimeoutError('fixed absolute/cumulative budget exhausted; no resume')
    try:
        guard();tic=time.monotonic();source=Path(P['old_root'])
        assert sha(HERE/'cohort.json')==P['cohort_sha256']
        for f,h in P['old_source_manifests'].items():assert sha(source/f)==h,('old manifest drift',f)
        for p,h in json.loads((source/'inherited_manifest.json').read_text()).items():guard();assert sha(p)==h,('inherited drift',p)
        cohort=json.loads((HERE/'cohort.json').read_text());assert len(cohort)==48 and sum(c['focal'] for c in cohort)==16
        if args.preflight:cohort=[c for c in cohort if c['family'] in P['preflight_families']];assert len(cohort)==3
        observations={c[k] for c in cohort for k in ['eta0','eta1']};families={c['family'] for c in cohort}
        for mf in ['input_manifest.json','prediction_cohort.json']:
            for p,h in json.loads((source/mf).read_text()).items():
                path=Path(p);selected=(path.parent.name=='observations' and path.stem in observations) or (path.parent.name=='truth' and path.stem in families) or (path.parent.name=='predictions' and path.stem.removesuffix('__sc-inr') in observations and path.stem.endswith('__sc-inr'))
                if selected:guard();assert sha(source/p)==h,('input drift',p)
        write(root/'cohort.json',cohort)
        write(root/'input_freeze.json',dict(epoch=time.time(),source_root=str(source),source_sha256=sha(HERE/'run.py'),protocol_sha256=sha(HERE/'protocol.json'),cohort_sha256=sha(HERE/'cohort.json'),run_cohort_sha256=sha(root/'cohort.json'),old_manifests=P['old_source_manifests'],families=len(cohort),observations=len(observations),preflight=args.preflight))
        phase['prepare']=time.monotonic()-tic;tic=time.monotonic();infer(source,root,cohort,guard);phase['inference']=time.monotonic()-tic
        tic=time.monotonic();score(source,root,cohort,guard);phase['score']=time.monotonic()-tic;guard();status='complete'
        write(root/'COMPLETE.json',dict(epoch=time.time(),rows_sha256=sha(root/'rows.json'),summary_sha256=sha(root/'summary.json')))
    except BaseException:(root/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF);write(root/'resources.json',dict(status=status,wall_seconds=time.time()-launch['start_epoch'],script_seconds=time.monotonic()-start,cpu_seconds=u.ru_utime+u.ru_stime,peak_rss_kib=u.ru_maxrss,phase_seconds=phase,CPU_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),GPU=0,training=0,end_epoch=time.time()))
if __name__=='__main__':main()
