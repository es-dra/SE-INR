"""CPU-only fixed linear development controls; truth stays closed until freeze."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
STATE = {}
DEADLINE = float('inf')
BEGAN_MONO = 0.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def journal(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def rss():
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(val if sys.platform == 'darwin' else val*1024)


def checkpoint():
    if time.time() >= DEADLINE or time.monotonic()-BEGAN_MONO >= 600:
        raise TimeoutError('Frozen absolute/cumulative CPU-phase wall budget exhausted')


def interrupted(signum, frame):
    raise TimeoutError('Interrupted by signal '+str(signum))


def parse_source(root):
    if root is None: return None
    candidates = [root] if root.is_file() else [root/'original-e1ff.py', root/'e1ff.py']
    source = next((p for p in candidates if p.is_file()), None)
    if source is None: raise RuntimeError('No archived original FF source found')
    definitions = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'gauss_B']
    if len(definitions)!=1: raise RuntimeError('Expected one original gauss_B definition')
    return dict(path=str(source), sha256=sha(source), ast=ast.dump(definitions[0],include_attributes=False))


def run(args):
    global DEADLINE, BEGAN_MONO
    if args.root.exists(): raise RuntimeError('New root required; no resume or overwrite')
    args.root.mkdir(parents=True)
    began=time.time();BEGAN_MONO=time.monotonic()
    DEADLINE=min(began+600,args.deadline_unix or float('inf'))
    STATE.update(stage='setup',began_unix=began,deadline_unix=DEADLINE,solved=0,scored=0)
    protocol=json.loads((HERE/'protocol.json').read_text())
    frozen=dict(started_unix=began,deadline_unix=DEADLINE,protocol_sha256=sha(HERE/'protocol.json'),source_sha256=sha(Path(__file__)),original_source=parse_source(args.original_source_root))
    save(args.root/'frozen.json',frozen)
    (args.root/'protocol.json').write_bytes((HERE/'protocol.json').read_bytes())
    # Imports occur inside the wall budget; CUDA visibility is disabled above.
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve, eigvalsh
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    input_root=args.input_root
    geometries=[im+'_ph'+''.join(map(str,ph)) for im,ph in itertools.product(protocol['cohort']['images'],protocol['cohort']['phase_sets'])]
    cases=[(g,s,m) for g,s,m in itertools.product(geometries,protocol['cohort']['sigma'],protocol['cohort']['methods'])]
    assert len(cases)==24
    yy,xx=np.meshgrid(np.arange(64),np.arange(64),indexing='ij')
    evaluation_xy=np.stack([xx,yy],-1).reshape(-1,2)
    target=(evaluation_xy+.5)/64
    crop_mask=((evaluation_xy[:,0]>=4)&(evaluation_xy[:,0]<60)&(evaluation_xy[:,1]>=4)&(evaluation_xy[:,1]<60))
    inputs={};datasets={};frequency={};geometry_metadata={}
    metadata_paths=[input_root/p for p in ['frozen.json','resources.json','summary.json'] if (input_root/p).is_file()]
    inputs.update({str(p):sha(p) for p in metadata_paths})
    STATE['stage']='solve'
    def features(coords,B):
        z=2*np.pi*coords@B.T
        return np.concatenate([np.cos(z),np.sin(z)],axis=1)/math.sqrt(256)
    def rbf(x,z,sigma):
        dist=np.maximum((x*x).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*x@z.T,0.)
        return np.exp(-2*np.pi**2*sigma**2*dist)
    def covariance(x,z,sigma,method,B):
        return rbf(x,z,sigma) if method=='rbf' else features(x,B)@features(z,B).T
    outputs=[];failures=[]
    for geometry,sigma,method in cases:
        checkpoint()
        cid=geometry+'_sigma'+str(sigma)+'_'+method
        started=time.monotonic();timings={}
        dest=args.root/'predictions'/cid
        dest.mkdir(parents=True,exist_ok=True)
        record=dict(id=cid,geometry=geometry,sigma=sigma,method=method,status='in_progress')
        try:
            checkpoint()
            obsfile=input_root/'observations'/(geometry+'.npz')
            if geometry not in datasets:
                inputs[str(obsfile)]=sha(obsfile)
                with np.load(obsfile,allow_pickle=False) as obs:
                    # NPZ is lazy; neither clean nor truth is accessed here.
                    y=np.asarray(obs['y'],dtype=np.float64)
                    coords=np.asarray(obs['coords'],dtype=np.float64)
                    xy=np.asarray(obs['xy'],dtype=np.int64)
                if y.shape!=(512,3) or coords.shape!=(512,2) or xy.shape!=(512,2): raise RuntimeError('Unexpected observation shapes')
                if np.any(xy<0) or np.any(xy>=64):raise RuntimeError('Observation index outside HR64')
                if not np.array_equal(coords,(xy+.5)/64):raise RuntimeError('Coordinate/index mismatch')
                if not np.isfinite(y).all() or not np.isfinite(coords).all():raise RuntimeError('Nonfinite input')
                datasets[geometry]=(y,coords,xy)
                unique_xy=np.unique(xy,axis=0)
                # Exact integer squared distances; the kernel itself is nonperiodic.
                delta=evaluation_xy[:,None,:]-unique_xy[None,:,:]
                nearest_squared=np.min(np.sum(delta*delta,axis=2),axis=1)
                squared_values,counts=np.unique(nearest_squared[crop_mask],return_counts=True)
                geometry_metadata[geometry]=dict(nearest_observed_shave4=dict(metric='Nonperiodic Euclidean distance to unique observation xy; exact squared pixel distance bins',n_target_pixels=int(crop_mask.sum()),histogram=[dict(squared_distance_pixels=int(d),distance_pixels=math.sqrt(int(d)),distance_coordinate_units=math.sqrt(int(d))/64,count=int(n)) for d,n in zip(squared_values,counts)]),interpretation='Horizontal periodic maxgap is only a geometric description; phase observation sets are nonnested and the kernel is nonperiodic. Integer shave4 phase01/02 targets can share nearest-distance histograms despite different continuous periodic maxgap; no scalar distance explanation of PSNR is claimed.')
                del delta
            y,coords,xy=datasets[geometry]
            bkey=(geometry,sigma)
            checkpoint_file=input_root/'fits'/(geometry+'_sigma'+str(sigma))/'checkpoint.pth'
            if bkey not in frequency:
                inputs[str(checkpoint_file)]=sha(checkpoint_file)
                payload=torch.load(checkpoint_file,map_location='cpu',weights_only=True)
                if payload['steps']!=2000:raise RuntimeError('Old FF checkpoint is incomplete')
                B=payload['model']['B'].detach().cpu().numpy().astype(np.float64)
                if B.shape!=(256,2) or not np.isfinite(B).all():raise RuntimeError('Invalid original frequency buffer')
                frequency[bkey]=B
                del payload
            B=frequency[bkey]
            begin=time.monotonic()
            K=covariance(coords,coords,sigma,method,B)
            K=(K+K.T)/2
            Kto=covariance(target,coords,sigma,method,B)
            S=K+1e-4*np.eye(512)
            timings['kernel_seconds']=time.monotonic()-begin
            checkpoint();begin=time.monotonic()
            eigen=eigvalsh(K,check_finite=True)
            if eigen[0] < -1e-8:raise RuntimeError('Kernel eigenvalue below frozen PSD tolerance')
            s_eigen=eigen+1e-4
            relative=eigen/max(float(eigen[-1]),1e-30)
            if s_eigen[0]<=0:raise RuntimeError('Nonpositive regularized system eigenvalue')
            timings['eigen_seconds']=time.monotonic()-begin
            checkpoint();begin=time.monotonic()
            factor=cho_factor(S,lower=True,check_finite=True)
            coefficients=cho_solve(factor,y-.5,check_finite=True)
            residual=float(np.linalg.norm(S@coefficients-(y-.5))/max(np.linalg.norm(y-.5),1e-30))
            if residual>1e-8:raise RuntimeError('Posterior coefficient residual exceeds frozen tolerance')
            timings['cholesky_and_coefficient_seconds']=time.monotonic()-begin
            checkpoint();begin=time.monotonic()
            W=cho_solve(factor,Kto.T,check_finite=True).T
            w_residual=float(np.linalg.norm(S@W.T-Kto.T)/max(np.linalg.norm(Kto.T),1e-30))
            if w_residual>1e-8:raise RuntimeError('Linear response residual exceeds frozen tolerance')
            raw=(.5+Kto@coefficients).T.reshape(3,64,64)
            clamp=np.clip(raw,0,1)
            if not np.isfinite(raw).all() or not np.isfinite(W).all():raise RuntimeError('Nonfinite predictions/response')
            timings['response_and_prediction_seconds']=time.monotonic()-begin
            checkpoint();begin=time.monotonic()
            diag=np.ones(4096) if method=='rbf' else (features(target,B)**2).sum(1)
            posterior=diag-np.einsum('ij,ij->i',W,Kto)
            if not np.isfinite(posterior).all() or posterior.min() < -1e-8:raise RuntimeError('Invalid model posterior variance')
            noise_variance_rows=1e-4*(W**2).sum(1)
            timings['posterior_variance_seconds']=time.monotonic()-begin
            coverage={}
            for axis,label in [(0,'horizontal'),(1,'vertical')]:
                positions=np.unique(xy[:,axis]);gaps=np.diff(np.r_[positions,positions[0]+64])
                coverage[label]=dict(unique_sites=len(positions),periodic_max_gap=int(gaps.max()))
            coverage['unique_observed_pixels']=len(np.unique(xy,axis=0));coverage['n_noisy_observations']=len(xy)
            coverage.update(geometry_metadata[geometry])
            begin=time.monotonic()
            np.savez_compressed(dest/'arrays.npz',unclamped=raw,clamped=clamp,linear_response=W,posterior_coefficients=coefficients,frequency_B=B,observation_xy=xy,evaluation_xy=evaluation_xy,crop_mask=crop_mask,posterior_variance=posterior,expected_noise_mse_by_pixel=noise_variance_rows,kernel_eigenvalues=eigen)
            timings['save_seconds']=time.monotonic()-begin
            checkpoint()
            record.update(status='complete',source_observations_sha256=inputs[str(obsfile)],source_ff_checkpoint_sha256=inputs[str(checkpoint_file)],arrays_sha256=sha(dest/'arrays.npz'),relative_true_residual=residual,linear_response_relative_residual=w_residual,condition_number=float(s_eigen[-1]/s_eigen[0]),min_cholesky_pivot=float(np.diag(factor[0]).min()),min_kernel_eigenvalue=float(eigen[0]),max_kernel_eigenvalue=float(eigen[-1]),kernel_ranks={str(t):int(np.sum(relative>t)) for t in protocol['solver']['kernel_rank_relative_thresholds']},coverage=coverage,model_implied_variance=dict(full_mean=float(posterior.mean()),shave4_mean=float(posterior[crop_mask].mean()),minimum=float(posterior.min()),negative_entries=int(np.sum(posterior<0))),stage_wall_seconds=timings,total_case_wall_seconds=time.monotonic()-started,peak_rss_bytes=rss())
            save(dest/'record.json',record);outputs.append(record)
            STATE['solved']=len(outputs)
        except Exception as error:
            record.update(status='budget_incomplete' if isinstance(error,TimeoutError) else 'failed',failure=dict(type=type(error).__name__,message=str(error)),stage_wall_seconds=timings,total_case_wall_seconds=time.monotonic()-started,peak_rss_bytes=rss())
            save(dest/'record.json',record);failures.append(record)
            journal(args.root/'solve_journal.jsonl',record)
            if isinstance(error,TimeoutError):raise
            continue
        journal(args.root/'solve_journal.jsonl',record)
        print('SOLVED',cid,flush=True)
    save(args.root/'input_hashes.json',inputs)
    if failures or len(outputs)!=24:
        save(args.root/'prediction_failures.json',failures)
        raise RuntimeError('Prediction cohort incomplete; target truth stays closed')
    checkpoint()
    cohort=dict(n_cases=24,records={r['id']:dict(record_sha256=sha(args.root/'predictions'/r['id']/'record.json'),arrays_sha256=r['arrays_sha256']) for r in outputs},frozen_before_truth_score_unix=time.time(),source_sha256=frozen['source_sha256'])
    save(args.root/'prediction_cohort.json',cohort)
    cohort_sha=sha(args.root/'prediction_cohort.json')
    STATE['stage']='score'
    scores=[];ff_scores={}
    def evaluate(pred,truth,mask):
        error=pred.reshape(3,-1)[:,mask]-truth.reshape(3,-1)[:,mask]
        mse=float(np.mean(error**2));return dict(mse=mse,psnr=-10*math.log10(max(mse,1e-30)))
    for record in outputs:
        checkpoint();started=time.monotonic()
        cid,geometry=record['id'],record['geometry']
        dest=args.root/'predictions'/cid
        if sha(dest/'arrays.npz')!=cohort['records'][cid]['arrays_sha256'] or sha(dest/'record.json')!=cohort['records'][cid]['record_sha256']:raise RuntimeError('Frozen prediction changed')
        obsfile=input_root/'observations'/(geometry+'.npz')
        if sha(obsfile)!=inputs[str(obsfile)]:raise RuntimeError('Observed archive changed')
        with np.load(obsfile,allow_pickle=False) as obs:
            clean=np.asarray(obs['clean'],dtype=np.float64)
            truth=np.asarray(obs['truth'],dtype=np.float64)
        y,coords,xy=datasets[geometry]
        if clean.shape!=(512,3) or truth.shape!=(3,64,64) or not np.isfinite(clean).all() or not np.isfinite(truth).all():raise RuntimeError('Invalid scoring truth/clean')
        if not np.allclose(clean,truth[:,xy[:,1],xy[:,0]].T,atol=0,rtol=0):raise RuntimeError('Clean observations do not match truth point samples')
        with np.load(dest/'arrays.npz',allow_pickle=False) as arrays:
            raw=arrays['unclamped'];clamped=arrays['clamped'];W=arrays['linear_response'];crop=arrays['crop_mask'];expected_noise=arrays['expected_noise_mse_by_pixel']
        bias=(.5+W@(clean-.5)).T-truth.reshape(3,-1)
        realized_noise=(W@(y-clean)).T
        if not np.allclose(raw.reshape(3,-1)-truth.reshape(3,-1),bias+realized_noise,rtol=1e-10,atol=1e-10):raise RuntimeError('Bias/noise reconstruction mismatch')
        risk={};metrics={}
        for region,mask in [('full',np.ones(4096,dtype=bool)),('shave4',crop)]:
            bias_mse=float(np.mean(bias[:,mask]**2));noise_mse=float(np.mean(realized_noise[:,mask]**2));cross=float(2*np.mean(bias[:,mask]*realized_noise[:,mask]));expected=float(expected_noise[mask].mean())
            risk[region]=dict(bias_squared_mse=bias_mse,expected_noise_mse=expected,expected_deterministic_raw_mse=bias_mse+expected,realized_noise_mse=noise_mse,realized_bias_noise_cross=cross,realized_decomposition_mse=bias_mse+noise_mse+cross)
            metrics[region]=dict(unclamped=evaluate(raw,truth,mask),clamped=evaluate(clamped,truth,mask))
            if abs(risk[region]['realized_decomposition_mse']-metrics[region]['unclamped']['mse'])>1e-10:raise RuntimeError('Realized risk identity mismatch')
        fkey=geometry+'_sigma'+str(record['sigma'])
        if fkey not in ff_scores:
            ffpath=input_root/'fits'/fkey/'predictions.npz';inputs[str(ffpath)]=sha(ffpath)
            with np.load(ffpath,allow_pickle=False) as ff:
                ffraw=np.asarray(ff['unclamped'],dtype=np.float64);ffclamp=np.asarray(ff['clamped'],dtype=np.float64)
            if ffraw.shape!=(3,64,64) or not np.isfinite(ffraw).all() or not np.allclose(ffclamp,np.clip(ffraw,0,1),atol=0,rtol=0):raise RuntimeError('Invalid saved FF predictions')
            ff_scores[fkey]={region:dict(unclamped=evaluate(ffraw,truth,mask),clamped=evaluate(ffclamp,truth,mask)) for region,mask in [('full',np.ones(4096,dtype=bool)),('shave4',crop)]}
        row=dict(id=cid,status='complete',prediction_cohort_sha256=cohort_sha,metrics=metrics,deterministic_risk=risk,old_ff_metrics=ff_scores[fkey],score_seconds=time.monotonic()-started,peak_rss_bytes=rss())
        scores.append(row);journal(args.root/'score_journal.jsonl',row);STATE['scored']=len(scores)
        checkpoint()
    checkpoint()
    save(args.root/'scores.json',scores);save(args.root/'old_ff_scores.json',ff_scores);save(args.root/'input_hashes.json',inputs)
    save(args.root/'summary.json',dict(status='complete',n_predictions=24,n_scored=24,cohort_sha256=cohort_sha,numerical_checks_pass=True,max_coefficient_residual=max(r['relative_true_residual'] for r in outputs),max_response_residual=max(r['linear_response_relative_residual'] for r in outputs),development_only=True,interpretation='Fixed same-observation linear controls and post-hoc FF comparisons only; model-implied variance does not establish natural-image GP validity or actual risk calibration. Deterministic raw risk identity conditions on known clean/truth, not an observation-only selector.'))
    checkpoint()
    STATE['stage']='complete'
    (args.root/'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--input-root',type=Path)
    parser.add_argument('--original-source-root',type=Path)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        ast.parse(Path(__file__).read_text());p=json.loads((HERE/'protocol.json').read_text())
        assert math.prod(len(p['cohort'][k]) for k in ['images','phase_sets','sigma','methods'])==24
        parse_source(args.original_source_root)
        print('CPU source/protocol check passed; no observation/truth/prediction arrays parsed; no torch/SciPy/CUDA loaded')
    else:
        if args.root is None or args.input_root is None:parser.error('--root and --input-root required')
        if args.root.exists():raise SystemExit('Refusing existing root')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        before=resource.getrusage(resource.RUSAGE_SELF)
        try:
            run(args)
        except Exception as error:
            if args.root.exists():save(args.root/'INCOMPLETE.json',dict(exception=type(error).__name__,message=str(error),time_unix=time.time(),state=STATE,journals='solve_journal.jsonl and score_journal.jsonl preserve completed records'))
            raise
        finally:
            if args.root.exists():
                after=resource.getrusage(resource.RUSAGE_SELF)
                save(args.root/'resources.json',dict(**STATE,ended_unix=time.time(),full_wall_seconds=time.time()-STATE.get('began_unix',time.time()),cumulative_cpu_phase_wall_seconds=time.monotonic()-BEGAN_MONO if BEGAN_MONO else 0.,process_user_cpu_seconds=after.ru_utime-before.ru_utime,process_system_cpu_seconds=after.ru_stime-before.ru_stime,peak_rss_bytes=rss(),cpu_threads=4,gpu_processes=0))
