"""Native-norm allocation intervention, full vs fixed-zero-cell ReLU decoding.
Retrospective fixed cohort. No training, new data, GPU or resume.
"""
import os,sys,json,time,argparse,hashlib,resource,traceback,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
helper=Path(P['helper_root'])
assert sha(helper/'run.py')==P['helper_sha256']
assert sha(helper/'protocol.json')==P['helper_protocol_sha256']
spec=importlib.util.spec_from_file_location('frozen_sinc_helper',helper/'run.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
np=H.np;torch=H.torch;F=H.F;models=H.models
write=H.write;maxabs=H.maxabs;rms=H.rms

def tangent(z,deltas,layers):
    h=z;vs=deltas;masks=[]
    for i,(w,b) in enumerate(layers):
        h=F.linear(h,w,b);vs=[F.linear(v,w,None) for v in vs]
        if i<3:
            mask=h>0;masks.append(mask);h=h.clamp_min(0);vs=[v*mask for v in vs]
    return h,vs,masks

def fixed(z,layers,masks):
    for i,(w,b) in enumerate(layers):
        z=F.linear(z,w,b)
        if i<3:z=z*masks[i]
    return z

def check_derivatives(u,dn,du,layers):
    out,vs,masks=tangent(u,[dn,du],layers);ans={}
    for name,d,v in zip(['native','uniform'],[dn,du],vs):
        _,jvp=torch.autograd.functional.jvp(lambda x:H.decode(x,layers),u,d,create_graph=False,strict=False)
        fd=fixed(u+d,layers,masks)-fixed(u,layers,masks)
        a=dict(jvp=maxabs(v-jvp),frozen_affine=maxabs(v-fd));assert max(a.values())<=P['numeric']['probe_atol'],a
        ans[name]=a
    return ans

def infer(source,root,cohort,guard,preflight):
    ck=torch.load(Path(P['model_root'])/'snapshot/checkpoints/sc-inr/epoch-last.pth',map_location='cpu',weights_only=True)
    assert ck['epoch']==P['checkpoint_epoch'];net=models.make(ck['model'],load_sd=True).cpu().eval()
    for p in net.parameters():p.requires_grad_(False)
    layers=H.layers_of(net);q=H.coords();indices=P['numeric']['query_indices'];journal=[]
    np.savez_compressed(root/'probes/weights.npz',**{f'{t}{i}':a.numpy() for i,wb in enumerate(layers) for t,a in zip(['w','b'],wb)})
    for i,c in enumerate(cohort):
        guard();tic=time.monotonic();record=dict(family=c['family'],observation=c['eta1'])
        try:
            y=np.load(source/'observations'/(c['eta1']+'.npy'));inp=torch.from_numpy(np.tile(y[None],(3,1,1)))[None];captured=[]
            def hook(module,args,out):captured.append(args[0].detach().clone())
            handle=net.imnet.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    net.gen_feat(inp*2-1)
                    p0=(net.query_rgb(q,torch.zeros_like(q))[0]*.5+.5).clone();assert len(captured)==4
                    U=torch.stack(captured);captured.clear()
                    pn=(net.query_rgb(q,torch.full_like(q,2/64))[0]*.5+.5).clone();assert len(captured)==4
                    zn=torch.stack(captured);captured.clear();W,weights,skip,omega=H.geometry_and_envelope(net,q)
            finally:handle.remove()
            feature_error=maxabs(zn-U*W);assert feature_error<=P['numeric']['feature_atol']
            old=np.load(source/'predictions'/(c['eta1']+'__sc-inr.npz'))
            endpoint=max(float(np.max(abs(old['point']-p0.numpy().reshape(64,64,3)))),float(np.max(abs(old['native']-pn.numpy().reshape(64,64,3)))))
            assert endpoint<=P['numeric']['endpoint_atol']
            U=U.double();W=W.double();weights=weights.double();skip=skip.double()
            dn=U*(W-1);nr=rms(dn);ur=rms(U);assert ur>0;t=nr/ur;du=-t*U
            norm_error=abs(rms(du)-nr);assert norm_error<=1e-12
            base=skip*.5+.5
            output={k:base.clone() for k in ['point','native','uniform']};dv={k:torch.zeros_like(base) for k in ['fixed_native','fixed_uniform']};checks=[]
            for branch in range(4):
                guard();out,vs,_=tangent(U[branch],[dn[branch],du[branch]],layers)
                output['point']+=weights[branch]*out*.5
                output['native']+=weights[branch]*H.decode(U[branch]*W[branch],layers)*.5
                output['uniform']+=weights[branch]*H.decode(U[branch]*(1-t),layers)*.5
                for key,v in zip(['fixed_native','fixed_uniform'],vs):dv[key]+=weights[branch]*v*.5
                if preflight or i==0:checks.append(dict(branch=branch,checks=check_derivatives(U[branch,indices],dn[branch,indices],du[branch,indices],layers)))
            for k,v in dv.items():output[k]=output['point']+v
            rebuild=max(maxabs(output['point']-p0.double()),maxabs(output['native']-pn.double()));assert rebuild<=P['numeric']['endpoint_atol']
            arrays={k:v.numpy().reshape(64,64,3) for k,v in output.items()};arrays.update(old_point=p0.numpy().reshape(64,64,3),old_native=pn.numpy().reshape(64,64,3))
            assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(root/'predictions'/(c['family']+'.npz'),**arrays)
            np.savez_compressed(root/'probes'/(c['family']+'.npz'),U=U[:,indices].numpy(),W=W[:,indices].numpy(),omega=omega.numpy(),weights=weights[:,indices].numpy(),skip=skip[indices].numpy(),indices=np.array(indices),rel_cell=np.array([.5,.5],dtype=np.float32),uniform_t=np.array(t),native_feature_rms=np.array(nr),U_rms=np.array(ur))
            record.update(status='complete',endpoint_float32_maxdiff=endpoint,feature_product_maxdiff=feature_error,rebuild_double_maxdiff=rebuild,native_feature_rms=nr,uniform_feature_rms=rms(du),norm_error=norm_error,U_rms=ur,uniform_t=t,W_min=float(W.min()),W_max=float(W.max()),uniform_weight=1-t,derivative_checks=checks)
        except BaseException:record.update(status='failed',traceback=traceback.format_exc());raise
        finally:record['seconds']=time.monotonic()-tic;journal.append(record);write(root/'inference_journal.json',journal)
        if i%4==0:print('INFER',i+1,len(cohort),round(record['seconds'],3),flush=True)
    manifest={str(f.relative_to(root)):sha(f) for d in ['predictions','probes'] for f in sorted((root/d).glob('*'))};write(root/'prediction_cohort.json',manifest)
    write(root/'prediction_freeze.json',dict(epoch=time.time(),manifest_sha256=sha(root/'prediction_cohort.json'),families=len(cohort),prediction_files=len(cohort),probe_files=len(cohort)+1))

def score(source,root,cohort,guard):
    freeze=json.loads((root/'prediction_freeze.json').read_text());assert sha(root/'prediction_cohort.json')==freeze['manifest_sha256']
    rows=[];tg=P['evaluation']['gain_tol'];ta=P['evaluation']['direction_tol'];sg=lambda x,t:1 if x>t else -1 if x< -t else 0
    for c in cohort:
        guard();truth=np.load(source/'truth'/(c['family']+'.npz'))['target'];z=np.load(root/'predictions'/(c['family']+'.npz'))
        for roi,(lo,hi) in {'center':(16,48),'full':(0,64)}.items():
            pp={k:H.planes(z[k],lo,hi) for k in z.files}
            for band,T in enumerate(H.planes(truth,lo,hi)):
                e=T-pp['point'][band];ref=float(np.mean(e*e));row=dict(**c,roi=roi,band=band,reference_mse=ref)
                for name in ['native','uniform','fixed_native','fixed_uniform']:
                    d=pp[name][band]-pp['point'][band];a=float(np.mean(e*d));b=float(np.mean(d*d));g=2*a-b;mse=float(np.mean((T-pp[name][band])**2))
                    assert abs(ref-mse-g)<P['numeric']['score_atol'];row.update({f'a_{name}':a,f'b_{name}':b,f'g_{name}':g,f'mse_{name}':mse})
                row['full_improvement']=row['mse_native']-row['mse_uniform'];row['fixed_improvement']=row['mse_fixed_native']-row['mse_fixed_uniform']
                row['full_rescued']=row['g_native']< -tg and row['g_uniform']>=-tg;row['fixed_rescued']=row['g_fixed_native']< -tg and row['g_fixed_uniform']>=-tg
                eo=T-pp['old_point'][band];do=pp['old_native'][band]-pp['old_point'][band];ao=float(np.mean(eo*do));go=float(2*ao-np.mean(do*do))
                row.update(old_a=ao,old_g=go,sensitive_direction=sg(ao,ta)!=sg(row['a_native'],ta),sensitive_gain=sg(go,tg)!=sg(row['g_native'],tg));rows.append(row)
    write(root/'rows.json',rows);groups={};selectors={'all32':lambda c:c['template'].startswith('mid_'),'focal16':lambda c:c['focal'],'nonflip16':lambda c:c['template'].startswith('mid_') and not c['focal'],'sham16':lambda c:c['template']=='control_low'}
    for name,sel in selectors.items():
        for roi in ['center','full']:
            for band in range(4):
                rr=[r for r in rows if sel(r) and r['roi']==roi and r['band']==band]
                if not rr:continue
                keys=['full_improvement','fixed_improvement']+[k+'_'+m for k in ['a','b','g','mse'] for m in ['native','uniform','fixed_native','fixed_uniform']]
                groups[f'{name}/{roi}/b{band}']=dict(n=len(rr),full_rescued=sum(r['full_rescued'] for r in rr),fixed_rescued=sum(r['fixed_rescued'] for r in rr),median={k:float(np.median([r[k] for r in rr])) for k in keys},mean={k:float(np.mean([r[k] for r in rr])) for k in keys},positive_direction={m:sum(r['a_'+m]>ta for r in rr) for m in ['native','uniform','fixed_native','fixed_uniform']},negative_direction={m:sum(r['a_'+m]< -ta for r in rr) for m in ['native','uniform','fixed_native','fixed_uniform']},positive_gain={m:sum(r['g_'+m]>tg for r in rr) for m in ['native','uniform','fixed_native','fixed_uniform']},negative_gain={m:sum(r['g_'+m]< -tg for r in rr) for m in ['native','uniform','fixed_native','fixed_uniform']},sensitive_direction=sum(r['sensitive_direction'] for r in rr),sensitive_gain=sum(r['sensitive_gain'] for r in rr))
    gates={}
    if 'focal16/center/b2' in groups and groups['focal16/center/b2']['n']==16:
        g=groups['focal16/center/b2'];rgb=groups['focal16/center/b0'];gates=dict(full_rescue_majority=g['full_rescued']>8,fixed_rescue_majority=g['fixed_rescued']>8,full_median_improvement=g['median']['full_improvement']>tg,fixed_median_improvement=g['median']['fixed_improvement']>tg,fullRGB_not_worse=rgb['median']['full_improvement']>=-tg)
    write(root/'summary.json',dict(groups=groups,gates=gates,allocation_explanation_supported=all(gates.values()) if gates else None,rows=len(rows),families=len(cohort),origin=P['origin'],limitations=P['limitations'],training=0,GPU=0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--launch',required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args();root=Path(args.out);root.mkdir()
    for d in ['predictions','probes']:(root/d).mkdir()
    launch=json.loads(Path(args.launch).read_text());write(root/'launch.json',launch);start=time.monotonic();budget=P['budget']['preflight_seconds'] if args.preflight else P['budget']['formal_seconds'];assert 0<launch['deadline_epoch']-launch['start_epoch']<=budget+1e-6
    status='failed';phase={}
    def guard():
        if time.time()>launch['deadline_epoch'] or time.monotonic()-start>budget:raise TimeoutError('fixed absolute/cumulative budget exhausted; no resume')
    try:
        guard();tic=time.monotonic();source=Path(P['old_root']);assert sha(HERE/'cohort.json')==P['cohort_sha256']
        for f,h in P['old_source_manifests'].items():assert sha(source/f)==h,('old manifest drift',f)
        for p,h in json.loads((source/'inherited_manifest.json').read_text()).items():guard();assert sha(p)==h,('inherited drift',p)
        cohort=json.loads((HERE/'cohort.json').read_text());assert len(cohort)==48 and sum(c['focal'] for c in cohort)==16
        if args.preflight:cohort=[c for c in cohort if c['family'] in P['preflight_families']];assert len(cohort)==3
        observations={c['eta1'] for c in cohort};families={c['family'] for c in cohort}
        for mf in ['input_manifest.json','prediction_cohort.json']:
            for p,h in json.loads((source/mf).read_text()).items():
                path=Path(p);selected=(path.parent.name=='observations' and path.stem in observations) or (path.parent.name=='truth' and path.stem in families) or (path.parent.name=='predictions' and path.stem.removesuffix('__sc-inr') in observations and path.stem.endswith('__sc-inr'))
                if selected:guard();assert sha(source/p)==h,('input drift',p)
        write(root/'cohort.json',cohort);write(root/'input_freeze.json',dict(epoch=time.time(),source_root=str(source),source_sha256=sha(HERE/'run.py'),protocol_sha256=sha(HERE/'protocol.json'),cohort_sha256=sha(HERE/'cohort.json'),run_cohort_sha256=sha(root/'cohort.json'),helper_sha256=sha(helper/'run.py'),helper_protocol_sha256=sha(helper/'protocol.json'),old_manifests=P['old_source_manifests'],families=len(cohort),observations=len(observations),preflight=args.preflight))
        phase['prepare']=time.monotonic()-tic;tic=time.monotonic();infer(source,root,cohort,guard,args.preflight);phase['inference']=time.monotonic()-tic
        tic=time.monotonic();score(source,root,cohort,guard);phase['score']=time.monotonic()-tic;guard();status='complete';write(root/'COMPLETE.json',dict(epoch=time.time(),rows_sha256=sha(root/'rows.json'),summary_sha256=sha(root/'summary.json')))
    except BaseException:(root/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF);write(root/'resources.json',dict(status=status,wall_seconds=time.time()-launch['start_epoch'],script_seconds=time.monotonic()-start,cpu_seconds=u.ru_utime+u.ru_stime,peak_rss_kib=u.ru_maxrss,phase_seconds=phase,CPU_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),GPU=0,training=0,end_epoch=time.time()))
if __name__=='__main__':main()
