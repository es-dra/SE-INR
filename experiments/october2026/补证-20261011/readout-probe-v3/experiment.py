"""Frozen inference-only diagnostics. No optimiser, gradient, or model selection.
Run from an isolated root containing snapshot/src (the inherited model code).
"""
import argparse, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text())
torch.backends.mkldnn.enabled = False  # preserve v1 NGC CPU segfault; use standard ATen fallback
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
sys.path.insert(0, str(HERE / 'snapshot' / 'src'))
import models

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def write(p, d):
    p.write_text(json.dumps(d, indent=2, allow_nan=False))

def field(coords, k, direction, phase, widths=(0., 0.)):
    ky, kx = np.array(direction, dtype=float) * k
    t = np.sinc(ky * widths[0]) * np.sinc(kx * widths[1])
    return .5 + .25 * t * np.cos(2 * np.pi * (ky*coords[:, 0] + kx*coords[:, 1]) + phase)

def coords(n, crop=None):
    v = (np.arange(n, dtype=float) + .5) / n
    if crop is not None: v = v[crop[0]:crop[1]]
    return np.stack(np.meshgrid(v, v, indexing='ij'), -1).reshape(-1, 2)

def make_cases(smoke=False):
    if smoke:
        return [dict(id='smoke', kind='synthetic', n=16, k=8, direction=[0,1], phase=np.pi/4)]
    cases=[]
    for n in P['input_sizes']:
        for k in P['frequencies']:
            for d in P['directions']:
                for pi, ph in enumerate(P['phases']):
                    cases.append(dict(id=f'n{n}_k{k}_d{d[0]}{d[1]}_p{pi}', kind='synthetic', n=n, k=k, direction=d, phase=ph, phase_index=pi))
    for s in [2,4]: cases.append(dict(id=f'barbara_x{s}',kind='natural', n=256//s, scale=s))
    return cases

def prepare(out, cases):
    obs = out/'observations'; truth=out/'truth'; obs.mkdir(); truth.mkdir()
    specs=[]
    for c in cases:
        if c['kind']=='synthetic':
            cc=coords(c['n']); y=field(cc,c['k'],c['direction'],c['phase'],(1/c['n'],1/c['n']))
            y=np.tile(y.reshape(c['n'],c['n'])[None],(3,1,1)).astype('float32')
            q=coords(64, P['query_crop']); np.save(truth/(c['id']+'.npy'),field(q,c['k'],c['direction'],c['phase']))
        else:
            fp=Path(P['barbara_path']); im=Image.open(fp).convert('RGB'); w,h=im.size
            assert min(w,h)>=256
            hr=np.array(im.crop(((w-256)//2,(h-256)//2,(w-256)//2+256,(h-256)//2+256)),dtype='float32').transpose(2,0,1)/255
            np.save(truth/(c['id']+'.npy'),hr)
            y=F.interpolate(torch.from_numpy(hr)[None],size=(c['n'],c['n']),mode='bicubic',antialias=True,align_corners=False).clamp(0,1)[0].numpy()
        np.save(obs/(c['id']+'.npy'),y)
        specs.append(c)
    write(out/'cases.json',specs)
    write(out/'input_manifest.json',{str(p.relative_to(out)):sha(p) for d in [obs,truth] for p in sorted(d.glob('*'))})

def load_model(name):
    cp=Path(P['checkpoint_root'])/name/'epoch-last.pth'
    ck=torch.load(cp,map_location='cpu',weights_only=True)
    assert ck['epoch']==1000
    m=models.make(ck['model'],load_sd=True).cpu().eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, dict(path=str(cp),sha256=sha(cp),epoch=ck['epoch'],registry=ck['model']['name'],args=ck['model']['args'],parameters=sum(p.numel() for p in m.parameters()))

def predict(out, cases, guard):
    pred=out/'predictions'; pred.mkdir(); costs=[]; ckmeta={}
    names=P['models'] if len(cases)>1 else P['models']
    net={}
    for name in names:
        net[name],ckmeta[name]=load_model(name)
    if len(cases)>1:
        for name in P['ablated_models']:
            net[name],ckmeta[name]=load_model(name)
    write(out/'checkpoint_manifest.json',ckmeta)
    query_count=0
    @torch.inference_mode()
    def query(m, q, wh):
        nonlocal query_count
        result=[]
        qt=torch.from_numpy((2*q-1).astype('float32'))[None]
        ct=torch.tensor(wh,dtype=torch.float32).view(1,1,2).expand_as(qt)*2
        for i in range(0,qt.shape[1],P['query_chunk']):
            guard(); z=m.query_rgb(qt[:,i:i+P['query_chunk']],ct[:,i:i+P['query_chunk']])[0].numpy()*.5+.5
            assert np.isfinite(z).all()
            result.append(z); query_count += len(z)
        return np.concatenate(result).astype('float32')
    @torch.inference_mode()
    def features(m,y): m.gen_feat(torch.from_numpy(y)[None]*2-1)
    for ci,c in enumerate(cases):
        guard(); t=time.monotonic(); y=np.load(out/'observations'/(c['id']+'.npy'))
        if c['kind']=='synthetic':
            q=coords(64,P['query_crop'])
            for name in names:
                m=net[name]; features(m,y); data={'point':query(m,q,(0,0))}
                for tag,fac in P['target_footprints'].items():
                    wh=np.array(fac)/64
                    data['native_'+tag]=query(m,q,wh)
                    for order in P['quadrature_orders']:
                        # Midpoint tensor quadrature of the FINAL point-limit RGB field.
                        v=(np.arange(order)+.5)/order-.5
                        offsets=np.stack(np.meshgrid(v*wh[0],v*wh[1],indexing='ij'),-1).reshape(-1,2)
                        nodes=(q[:,None,:]+offsets[None,:,:]).reshape(-1,2)
                        data[f'quad{order}_'+tag]=query(m,nodes,(0,0)).reshape(len(q),order*order,3).mean(1)
                np.savez_compressed(pred/(c['id']+'__'+name+'.npz'),**data)
        else:
            q=coords(256)
            for name,m in net.items():
                features(m,y)
                data={'native':query(m,q,(1/256,1/256))}
                if name in names: data['point']=query(m,q,(0,0))
                np.savez_compressed(pred/(c['id']+'__'+name+'.npz'),**data)
        costs.append(dict(case=c['id'],seconds=time.monotonic()-t,query_rgb_pixels_cumulative=query_count))
        write(out/'inference_journal.json',costs)
        print('PRED',ci+1,len(cases),c['id'],round(costs[-1]['seconds'],2),flush=True)
    cohort={str(p.relative_to(out)):sha(p) for p in sorted(pred.glob('*'))}
    write(out/'prediction_cohort.json',cohort)
    # Every prediction is saved and frozen before scoring starts.
    write(out/'prediction_freeze.json',dict(cohort_sha256=sha(out/'prediction_cohort.json'),files=len(cohort),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),query_rgb_pixels=query_count))

def mse(a,b): return float(np.mean((a-b)**2))

def safe_rho(x,y):
    from scipy.stats import spearmanr
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0: return None
    v=float(spearmanr(x,y).statistic)
    return v if np.isfinite(v) else None

def score(out,cases):
    from scipy.stats import spearmanr
    for n,h in json.loads((out/'prediction_cohort.json').read_text()).items(): assert sha(out/n)==h
    rows=[]; natural=[]
    for c in cases:
        if c['kind']=='synthetic':
            q=coords(64,P['query_crop']); point_truth=np.load(out/'truth'/(c['id']+'.npy'))[:,None]
            for name in P['models']:
                z=np.load(out/'predictions'/(c['id']+'__'+name+'.npz')); p0=z['point'].astype('float64')
                for tag,fac in P['target_footprints'].items():
                    wh=np.array(fac)/64; ky,kx=np.array(c['direction'])*c['k']; T=float(np.sinc(ky*wh[0])*np.sinc(kx*wh[1]))
                    target=field(q,c['k'],c['direction'],c['phase'],wh)[:,None]
                    S=float(np.mean((target-.5)**2)); assert S>1e-8
                    ph=z['native_'+tag].astype('float64'); q4=z['quad4_'+tag].astype('float64'); q8=z['quad8_'+tag].astype('float64')
                    ideal_scaled=.5+T*(p0-.5)
                    e=mse(p0,target); en=mse(ph,target); eq=mse(q8,target)
                    d=ph-p0; error=p0-target
                    energy=float(np.mean(d*d))/S; cross=2*float(np.mean(error*d))/S
                    row=dict(case=c['id'],model=name,n=c['n'],k=c['k'],direction=c['direction'],phase_index=c.get('phase_index',0),footprint=tag,band=1 if c['k']<8 else 2 if c['k']<16 else 3,T=T,signal_energy=S,native_mse=en,point_mse=e,quadrature_mse=eq,gain=(e-en)/S,footprint_score=(1-T)**2,scaling_pred_gain=(e-mse(ideal_scaled,target))/S,quadrature_pred_gain=(e-eq)/S,commutation_mse=mse(ph,q8),quadrature_refinement_mse=mse(q8,q4),change_energy=energy,alignment=cross,identity_residual=(e-en)/S+energy+cross,clamped_native_mse=mse(np.clip(ph,0,1),target),out_of_range_fraction=float(np.mean((ph<0)|(ph>1))))
                    row['quadrature_resolved'] = row['quadrature_refinement_mse'] <= 1e-5 and row['quadrature_refinement_mse'] <= .1 * row['commutation_mse']
                    rows.append(row)
        else:
            truth=np.load(out/'truth'/(c['id']+'.npy')).transpose(1,2,0)
            f=np.fft.fftfreq(256); radius=np.maximum(abs(f[:,None]),abs(f[None,:])); sig=truth.mean(2); X=np.fft.fft2(sig-sig.mean())
            for name,abl in zip(P['models'],P['ablated_models']):
                z=np.load(out/'predictions'/(c['id']+'__'+name+'.npz')); a=np.load(out/'predictions'/(c['id']+'__'+abl+'.npz'))
                full=z['native'].reshape(256,256,3).astype('float64'); point=z['point'].reshape(256,256,3).astype('float64'); ab=a['native'].reshape(256,256,3).astype('float64')
                for clamp in [False,True]:
                    fg=np.clip(full,0,1) if clamp else full; pg=np.clip(point,0,1) if clamp else point; ag=np.clip(ab,0,1) if clamp else ab
                    E=np.fft.fft2(sig-fg.mean(2)); Ep=np.fft.fft2(sig-pg.mean(2)); Ea=np.fft.fft2(sig-ag.mean(2)); D=np.fft.fft2(fg.mean(2)-pg.mean(2))
                    for b,(lo,hi) in enumerate([(0,.125),(.125,.25),(.25,.5001)],1):
                        mask=(radius>=lo)&(radius<hi)&(radius>0); S=float((abs(X[mask])**2).sum())
                        gain=float(((abs(Ep[mask])**2-abs(E[mask])**2).sum())/S); energy=float((abs(D[mask])**2).sum())/S; align=2*float((Ep[mask].conj()*D[mask]).real.sum())/S
                        natural.append(dict(case=c['id'],model=name,band=b,clamped=clamp,paired_retrained_gain=float(((abs(Ea[mask])**2-abs(E[mask])**2).sum())/S),fixed_checkpoint_gain=gain,change_energy=energy,alignment=align,identity_residual=gain+energy-align))
    write(out/'synthetic_rows.json',rows); write(out/'natural_rows.json',natural)
    groups=[]
    for name in P['models']:
        for n in P['input_sizes']:
            cal=[r for r in rows if r['model']==name and r['n']==n and r['phase_index']==0]
            test=[r for r in rows if r['model']==name and r['n']==n and r['phase_index']==1]
            if not test: continue
            fit=np.linalg.lstsq(np.array([[1,r['footprint_score']] for r in cal]),np.array([r['gain'] for r in cal]),rcond=None)[0]
            for r in test:
                r['band_pred_gain']=float(np.median([a['gain'] for a in cal if a['band']==r['band']]))
                r['footprint_fit_pred_gain']=float(fit@[1,r['footprint_score']])
            active=[r for r in test if abs(r['gain'])>P['sign_margin']]; negative=[r for r in active if r['gain']<0]
            stats={}
            for key in ['band_pred_gain','footprint_fit_pred_gain','scaling_pred_gain','quadrature_pred_gain']:
                predv=np.array([r[key] for r in test]); actual=np.array([r['gain'] for r in test]); rho=spearmanr(predv,actual).statistic
                stats[key]=dict(mae=float(np.mean(abs(predv-actual))),spearman=float(rho) if np.isfinite(rho) else None,sign_accuracy=float(np.mean([np.sign(r[key])==np.sign(r['gain']) for r in active])) if active else None,negative_recall=float(np.mean([r[key]<0 for r in negative])) if negative else None,within_band_spearman={str(b):safe_rho([r[key] for r in test if r['band']==b],[r['gain'] for r in test if r['band']==b]) if key!='band_pred_gain' else None for b in [1,2,3]})
            g=dict(model=name,n=n,test_rows=len(test),active_sign_rows=len(active),negative_rows=len(negative),predictors=stats,commutation_median=float(np.median([r['commutation_mse']/r['signal_energy'] for r in test])),quadrature_refinement_max=float(max(r['quadrature_refinement_mse'] for r in test)))
            g['quadrature_resolved_rows']=sum(r['quadrature_resolved'] for r in test)
            g['candidate_gate']={}
            for key in ['scaling_pred_gain','quadrature_pred_gain']:
                st=stats[key]; simple=min(stats['band_pred_gain']['mae'],stats['footprint_fit_pred_gain']['mae'])
                g['candidate_gate'][key] = bool(st['mae'] <= .8*simple and st['sign_accuracy'] is not None and st['sign_accuracy']>=.75 and len(negative)>=8 and st['negative_recall']>=.75 and (key!='quadrature_pred_gain' or all(r['quadrature_resolved'] for r in test)))
            groups.append(g)
    write(out/'synthetic_rows.json',rows)
    write(out/'summary.json',dict(synthetic_rows=len(rows),natural_rows=len(natural),groups=groups,max_identity_residual=max([abs(r['identity_residual']) for r in rows+natural] or [0]),scope=P['scope'],model_training_runs=0))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--launch-record'); a=ap.parse_args()
    out=Path(a.out); assert not out.exists(); out.mkdir(parents=True)
    start=time.monotonic(); limit=120 if a.smoke else P['wall_budget_seconds']
    launch=json.loads(Path(a.launch_record).read_text()) if a.launch_record else dict(start_epoch=time.time(),absolute_deadline_epoch=time.time()+limit)
    start_epoch=launch['start_epoch']; deadline=launch['absolute_deadline_epoch']
    def guard():
        if time.monotonic()-start>limit or time.time()>deadline: raise TimeoutError('fixed wall budget exceeded')
    source={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.suffix in ['.py','.json']}
    write(out/'frozen.json',dict(source_sha256=source,protocol_sha256=sha(HERE/'protocol.json'),start_epoch=start_epoch,absolute_deadline_epoch=deadline,wall_and_cumulative_phase_budget_seconds=limit,device='cpu',threads=4,nice=os.getpriority(os.PRIO_PROCESS,0),smoke=a.smoke,torch=torch.__version__,models_import=str(models.__file__)))
    cases=make_cases(a.smoke); phases=[]; status='failed'
    try:
        for phase in ['prepare','predict','score']:
            guard(); t=time.monotonic()
            if phase=='prepare': prepare(out,cases)
            elif phase=='predict': predict(out,cases,guard)
            else: score(out,cases)
            phases.append(dict(phase=phase,seconds=time.monotonic()-t)); write(out/'phase_costs.json',phases)
        guard(); (out/'COMPLETE').write_text('complete; inference only\n'); status='complete'
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        r=resource.getrusage(resource.RUSAGE_SELF)
        write(out/'resources.json',dict(status=status,wall_seconds=time.time()-start_epoch,in_script_wall_seconds=time.monotonic()-start,cpu_user_seconds=r.ru_utime,cpu_system_seconds=r.ru_stime,maxrss_kib=r.ru_maxrss,phases=phases,training_runs=0,gpu_processes=0))

if __name__=='__main__': main()
