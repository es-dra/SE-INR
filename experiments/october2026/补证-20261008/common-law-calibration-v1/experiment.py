"""Classic common-law risk calibration; frozen fixed-W CPU development diagnostic."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
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

HERE=Path(__file__).resolve().parent
STATE={}
START=0.
DEADLINE=float('inf')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def journal(path,value):
    with path.open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def rss():
    r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform=='darwin' else r*1024)
def check():
    if time.time()>=DEADLINE or time.monotonic()-START>=600:raise TimeoutError('Fixed absolute/cumulative600s CPU wall budget exhausted')
def interrupted(signum,frame):raise TimeoutError('Signal '+str(signum))

def run(args):
    global START,DEADLINE
    if args.root.exists():raise RuntimeError('Newroot required; no resume or overwrite')
    args.root.mkdir(parents=True)
    START=time.monotonic();began=time.time();DEADLINE=min(began+600,args.deadline_unix or float('inf'))
    STATE.update(stage='setup',began_unix=began,deadline_unix=DEADLINE,inputs_verified=0,calibration_rows=0,scored=0)
    os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    p=json.loads((HERE/'protocol.json').read_text())
    identities=dict(source_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'protocol.json'))
    save(args.root/'source_frozen.json',identities)
    (args.root/'experiment.py').write_bytes(Path(__file__).read_bytes())
    (args.root/'protocol.json').write_bytes((HERE/'protocol.json').read_bytes())
    check()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(4);torch.set_num_interop_threads(4)
    linear=args.linear_root
    if not (linear/'COMPLETE').is_file():raise RuntimeError('Complete old linear cohort required')
    cohort=json.loads((linear/'prediction_cohort.json').read_text())
    old_inputs=json.loads((linear/'input_hashes.json').read_text())
    hashes={str(linear/name):sha(linear/name) for name in ['prediction_cohort.json','input_hashes.json','frozen.json','protocol.json','COMPLETE']}
    old_frozen=json.loads((linear/'frozen.json').read_text())
    if old_frozen['protocol_sha256']!=hashes[str(linear/'protocol.json')] or cohort['source_sha256']!=old_frozen['source_sha256']:raise RuntimeError('Old linear source/protocol identity conflict')
    operators={};records={};timings={};equivalence=[]
    arms=[(m,s,ph) for m,s,ph in itertools.product(p['cohort']['methods'],p['cohort']['sigma'],p['cohort']['phases'])]
    def armid(arm):m,s,ph=arm;return 'ph'+ph+'_sigma'+str(s)+'_'+m
    STATE['stage']='verify_operators';start=time.monotonic()
    for im in p['cohort']['development_images']:
        for arm in arms:
            check();cid=im+'_'+armid(arm);STATE['current']=cid;entry=cohort['records'][cid]
            folder=linear/'predictions'/cid
            for name,key in [('arrays.npz','arrays_sha256'),('record.json','record_sha256')]:
                path=folder/name;digest=sha(path)
                if digest!=entry[key]:raise RuntimeError('Old frozen input SHA mismatch: '+cid+'/'+name)
                hashes[str(path)]=digest
            # Lazy NPZ reads: no saved prediction, Y, clean/truth or posterior variance.
            with np.load(folder/'arrays.npz',allow_pickle=False) as a:
                item={key:a[key] for key in p['operator_gate']['keys_before_freeze']}
            W=item['linear_response'];xy=item['observation_xy'];evalxy=item['evaluation_xy'];mask=item['crop_mask'];B=item['frequency_B']
            if W.dtype!=np.float64 or W.shape!=(4096,512) or xy.shape!=(512,2) or evalxy.shape!=(4096,2) or mask.shape!=(4096,) or B.shape!=(256,2):raise RuntimeError('Incomplete operator arrays')
            if not all(np.isfinite(v).all() for v in [W,xy,evalxy,B]) or mask.dtype!=np.bool_ or mask.sum()!=3136:raise RuntimeError('Invalid/ nonfinite operator')
            yy,xx=np.meshgrid(np.arange(64),np.arange(64),indexing='ij');expected=np.stack([xx,yy],-1).reshape(-1,2)
            expectedmask=((expected[:,0]>=4)&(expected[:,0]<60)&(expected[:,1]>=4)&(expected[:,1]<60))
            if not np.array_equal(evalxy,expected) or not np.array_equal(mask,expectedmask) or np.any(xy<0) or np.any(xy>=64):raise RuntimeError('Index/crop mismatch')
            if im=='0801':operators[arm]=item
            else:
                for key in item:
                    delta=float(np.max(np.abs(item[key].astype(float)-operators[arm][key].astype(float))))
                    exact=np.array_equal(item[key],operators[arm][key])
                    if key in ['observation_xy','evaluation_xy','crop_mask'] and not exact:raise RuntimeError('Geometry differs across images')
                    byte_identical=item[key].dtype==operators[arm][key].dtype and item[key].shape==operators[arm][key].shape and item[key].tobytes()==operators[arm][key].tobytes()
                    if key!='linear_response' and not byte_identical:raise RuntimeError('B/geometry bytes differ across images')
                    if delta>1e-14:raise RuntimeError('W/B consistency threshold exceeded')
                    equivalence.append(dict(arm=armid(arm),key=key,byte_identical=byte_identical,max_absolute_difference=delta))
            records[(im,arm)]=folder/'record.json'
            STATE['inputs_verified']+=1
            journal(args.root/'input_journal.jsonl',dict(id=cid,status='verified',arrays_sha256=entry['arrays_sha256'],record_sha256=entry['record_sha256']))
    if len(cohort['records'])!=24 or STATE['inputs_verified']!=24:raise RuntimeError('Expected complete24-case operator cohort')
    timings['operator_verification_seconds']=time.monotonic()-start
    save(args.root/'operator_equivalence.json',equivalence)
    reference_hashes={armid(arm):dict(W_sha256=hashlib.sha256(op['linear_response'].tobytes()).hexdigest(),B_sha256=hashlib.sha256(op['frequency_B'].tobytes()).hexdigest(),xy_sha256=hashlib.sha256(op['observation_xy'].tobytes()).hexdigest()) for arm,op in operators.items()}
    noise={arm:float(1e-4*np.sum(op['linear_response'][op['crop_mask']]**2)/3136) for arm,op in operators.items()}
    data=args.data_root/'DIV2K_train_HR' if (args.data_root/'DIV2K_train_HR').is_dir() else args.data_root
    STATE['stage']='calibration';rows=[];ztruth=[];zobs={arm:[] for arm in arms};cal_hashes={};pretime=0.;armtime={armid(a):0. for a in arms}
    for im in p['calibration_images']:
        check();STATE['current']='calibration_'+im;start=time.monotonic();path=data/(im+'.png');digest=sha(path);cal_hashes[im]=dict(path=str(path),sha256=digest)
        with Image.open(path) as image:
            image=image.convert('RGB');w,h=image.size
            if min(w,h)<128:raise RuntimeError('Calibration image too small: '+im)
            box=((w-128)//2,(h-128)//2,(w-128)//2+128,(h-128)//2+128)
            native=np.asarray(image.crop(box),dtype=np.float32)/255.
        truth=F.avg_pool2d(torch.from_numpy(native).permute(2,0,1)[None],2)[0].numpy().astype(np.float64)
        if sha(path)!=digest:raise RuntimeError('Calibration source changed during read')
        if truth.shape!=(3,64,64) or not np.isfinite(truth).all():raise RuntimeError('Invalid calibration image')
        target=truth.reshape(3,-1)[:,operators[arms[0]]['crop_mask']].T
        ztruth.append(target-.5);preprocess_seconds=time.monotonic()-start;pretime+=preprocess_seconds
        journal(args.root/'preprocess_journal.jsonl',dict(image=im,source_sha256=digest,crop=box,preprocess_seconds=preprocess_seconds,pooled_float64_sha256=hashlib.sha256(truth.tobytes()).hexdigest()))
        for arm in arms:
            check();STATE['current']='calibration_'+im+'_'+armid(arm);start=time.monotonic();op=operators[arm];xy=op['observation_xy'];clean=truth[:,xy[:,1],xy[:,0]].T
            Z=clean-.5;error=op['linear_response'][op['crop_mask']]@Z-(target-.5)
            bias=float(np.mean(error**2));total=bias+noise[arm]
            if not np.isfinite(total):raise RuntimeError('Nonfinite calibration risk')
            zobs[arm].append(Z);armtime[armid(arm)]+=time.monotonic()-start
            row=dict(image=im,arm=armid(arm),bias_mse=bias,noise_mse=noise[arm],total_mse=total,crop=box,seconds=time.monotonic()-start)
            rows.append(row);journal(args.root/'calibration_journal.jsonl',row);STATE['calibration_rows']=len(rows)
            check()
    if len(rows)!=384:raise RuntimeError('Calibration cohort incomplete')
    timings.update(calibration_preprocess_seconds=pretime,calibration_arm_seconds=armtime)
    calrisk={arm:float(np.mean([r['total_mse'] for r in rows if r['arm']==armid(arm)])) for arm in arms}
    start=time.monotonic();Zt=np.concatenate(ztruth,axis=1);moment=[]
    for arm in arms:
        check();op=operators[arm];Zo=np.concatenate(zobs[arm],axis=1);residual=op['linear_response'][op['crop_mask']]@Zo-Zt
        value=float(np.sum(residual**2)/(32*3*3136)+noise[arm]);delta=abs(value-calrisk[arm])
        if not np.isfinite(value) or delta>1e-12:raise RuntimeError('Low-rank second-moment equivalence failed')
        moment.append(dict(arm=armid(arm),residual_matrix_shape=list(residual.shape),equivalent_risk=value,absolute_difference=delta))
    timings['second_moment_seconds']=time.monotonic()-start
    start=time.monotonic()
    groups={'phase'+ph:[a for a in arms if a[2]==ph] for ph in p['cohort']['phases']}
    groups.update({m+'_sigma'+str(s):[a for a in arms if a[:2]==(m,s)] for m,s in itertools.product(p['cohort']['methods'],p['cohort']['sigma'])});groups['global']=arms
    def choose(values,candidates):
        minimum=min(values[a] for a in candidates)
        return sorted(a for a in candidates if values[a]<=minimum+1e-12)[0]
    recommendations={key:dict(chosen=armid(choose(calrisk,candidates)),risk=calrisk[choose(calrisk,candidates)],candidates=[armid(a) for a in sorted(candidates)]) for key,candidates in groups.items()}
    timings['recommendation_seconds']=time.monotonic()-start
    save(args.root/'calibration_rows.json',rows);save(args.root/'calibration_risks.json',{armid(a):calrisk[a] for a in arms});save(args.root/'second_moment_check.json',moment);save(args.root/'recommendations.json',recommendations)
    check();STATE['stage']='freeze';start=time.monotonic()
    for entry in cal_hashes.values():
        check()
        if sha(Path(entry['path']))!=entry['sha256']:raise RuntimeError('Calibration source changed before freeze')
    if sha(Path(__file__))!=identities['source_sha256'] or sha(HERE/'protocol.json')!=identities['protocol_sha256']:raise RuntimeError('Source/protocol changed during calibration')
    frozen_files={name:sha(args.root/name) for name in ['experiment.py','source_frozen.json','protocol.json','operator_equivalence.json','calibration_rows.json','calibration_risks.json','second_moment_check.json','recommendations.json']}
    freeze=dict(identities=identities,all24_input_sha256=hashes,reference12_operator_hashes=reference_hashes,calibration32_source_hashes=cal_hashes,frozen_files=frozen_files,frozen_before_development_score_unix=time.time(),n_calibration_rows=384,n_unique_operators=12,recommendations=recommendations)
    save(args.root/'recommendation_cohort.json',freeze);freeze_sha=sha(args.root/'recommendation_cohort.json')
    for name,digest in frozen_files.items():
        if sha(args.root/name)!=digest:raise RuntimeError('Frozen calibration output changed')
    for path,digest in hashes.items():
        if sha(Path(path))!=digest:raise RuntimeError('Frozen operator input changed')
    timings['hash_freeze_seconds']=time.monotonic()-start
    check();STATE['stage']='postfreeze_score';start=time.monotonic()
    # Development scoring starts here. No development Y/truth/score metadata above.
    scorefile=linear/'scores.json';score_rows=json.loads(scorefile.read_text());posthashes={str(scorefile):sha(scorefile)}
    oldscores={r['id']:r for r in score_rows}
    if len(oldscores)!=24:raise RuntimeError('Missing old control scores')
    posterior={};expected={};comparisons=[];pairs=[];scored=[];baselines=[]
    def sign(v):return 0 if abs(v)<=1e-12 else (1 if v>0 else -1)
    for im in p['cohort']['development_images']:
        for ph in p['cohort']['phases']:
            check();geometry=im+'_ph'+ph;suffix='/observations/'+geometry+'.npz'
            matches=[(path,digest) for path,digest in old_inputs.items() if path.endswith(suffix)]
            if len(matches)!=1:raise RuntimeError('Cannot identify frozen development observation archive')
            original,digest=matches[0];local=linear.parent/'inr-phase-development-run-v1'/'observations'/(geometry+'.npz')
            obsfile=Path(original) if Path(original).is_file() else local
            if sha(obsfile)!=digest:raise RuntimeError('Development archive SHA mismatch')
            posthashes[str(obsfile)]=digest
            with np.load(obsfile,allow_pickle=False) as obs:clean=np.asarray(obs['clean'],dtype=np.float64);truth=np.asarray(obs['truth'],dtype=np.float64)
            if clean.shape!=(512,3) or truth.shape!=(3,64,64) or not np.isfinite(clean).all() or not np.isfinite(truth).all():raise RuntimeError('Invalid development clean/truth')
            for arm in groups['phase'+ph]:
                check();STATE['current']=im+'_'+armid(arm);op=operators[arm];xy=op['observation_xy'];mask=op['crop_mask']
                if not np.array_equal(clean,truth[:,xy[:,1],xy[:,0]].T):raise RuntimeError('Development geometry mismatch')
                error=op['linear_response'][mask]@(clean-.5)-(truth.reshape(3,-1)[:,mask].T-.5)
                risk=float(np.mean(error**2)+noise[arm]);expected[(im,arm)]=risk
                record=json.loads(records[(im,arm)].read_text());posterior[(im,arm)]=record['model_implied_variance']['shave4_mean']
                old=oldscores[im+'_'+armid(arm)]
                delta=abs(risk-old['deterministic_risk']['shave4']['expected_deterministic_raw_mse'])
                if not np.isfinite(risk) or not np.isfinite(posterior[(im,arm)]) or delta>1e-12:raise RuntimeError('Postfreeze expected-risk consistency failure')
                row=dict(image=im,arm=armid(arm),expected_raw_mse=risk,recomputed_vs_old_difference=delta,realized_clamped=old['metrics']['shave4']['clamped'],own_prior_posterior=posterior[(im,arm)],recommendation_cohort_sha256=freeze_sha)
                scored.append(row);journal(args.root/'score_journal.jsonl',row);STATE['scored']=len(scored);check()
        vals={a:expected[(im,a)] for a in arms}
        for group,candidates in groups.items():
            chosen=choose(calrisk,candidates);oracle=choose(vals,candidates)
            comparisons.append(dict(image=im,group=group,calibration_chosen=armid(chosen),oracle_chosen=armid(oracle),same_choice=chosen==oracle,chosen_expected_raw_mse=vals[chosen],minimum_expected_raw_mse=min(vals[a] for a in candidates),expected_raw_regret=vals[chosen]-min(vals[a] for a in candidates)))
        for a,b in itertools.combinations(sorted(arms),2):
            gap=calrisk[a]-calrisk[b];actual=vals[a]-vals[b]
            pairs.append(dict(image=im,arm_a=armid(a),arm_b=armid(b),calibration_gap=gap,actual_expected_raw_gap=actual,calibration_sign=sign(gap),actual_sign=sign(actual),opposing=sign(gap)*sign(actual)==-1))
        for ph in p['cohort']['phases']:
            candidates=groups['phase'+ph];postvalues={a:posterior[(im,a)] for a in candidates};chosen=choose(postvalues,candidates);fixed=('rbf',3,ph);minimum=min(vals[a] for a in candidates)
            baselines.append(dict(image=im,phase=ph,fixed_rbf_sigma3=dict(arm=armid(fixed),expected_raw_mse=vals[fixed],expected_raw_regret=vals[fixed]-minimum),ownprior_postfreeze=dict(arm=armid(chosen),posterior=postvalues[chosen],expected_raw_mse=vals[chosen],expected_raw_regret=vals[chosen]-minimum)))
        check()
    if len(scored)!=24 or len(pairs)!=132 or len(comparisons)!=16:raise RuntimeError('Postfreeze cohort incomplete')
    timings['postfreeze_score_seconds']=time.monotonic()-start
    check();save(args.root/'postfreeze_scores.json',scored);save(args.root/'recommendation_comparisons.json',comparisons);save(args.root/'pair_gaps.json',pairs);save(args.root/'opposing_pairs.json',[r for r in pairs if r['opposing']]);save(args.root/'baselines.json',baselines);save(args.root/'postfreeze_input_hashes.json',posthashes)
    save(args.root/'summary.json',dict(status='complete',development_only=True,n_calibration_images=32,n_unique_operators=12,n_operator_inputs=24,n_calibration_image_arm_rows=384,n_development_scores=24,n_recommendation_comparisons=16,n_pair_comparisons=132,n_opposing_pairs=sum(r['opposing'] for r in pairs),recommendation_matches=sum(r['same_choice'] for r in comparisons),recommendation_cohort_sha256=freeze_sha,stage_wall_seconds=timings,interpretation='Fixed finite-ensemble calibration-static recommendations and retrospective expected-raw/realized-clamped scores only. No adaptive selector or generalization conclusion.'))
    check();STATE['stage']='complete';(args.root/'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--linear-root',type=Path);parser.add_argument('--data-root',type=Path);parser.add_argument('--deadline-unix',type=float);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        ast.parse(Path(__file__).read_text());p=json.loads((HERE/'protocol.json').read_text())
        assert len(p['calibration_images'])==32 and len(set(p['calibration_images']))==32
        assert math.prod(len(p['cohort'][k]) for k in ['phases','sigma','methods'])==12
        print('AST/protocol checks passed; no torch/numpy/CUDA loaded, no image/operator/development truth read')
    else:
        if any(v is None for v in [args.root,args.linear_root,args.data_root]):parser.error('--root --linear-root --data-root required')
        if args.root.exists():raise SystemExit('Refusing existing root')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);before=resource.getrusage(resource.RUSAGE_SELF)
        try:run(args)
        except Exception as error:
            if args.root.exists():save(args.root/'INCOMPLETE.json',dict(exception=type(error).__name__,message=str(error),state=STATE,time_unix=time.time(),partial_evidence='input/calibration/score journals retain completed rows; no continuation'))
            raise
        finally:
            if args.root.exists():
                after=resource.getrusage(resource.RUSAGE_SELF);save(args.root/'resources.json',dict(**STATE,ended_unix=time.time(),full_wall_seconds=time.monotonic()-START if START else 0,cumulative_cpu_phase_wall_seconds=time.monotonic()-START if START else 0,process_user_cpu_seconds=after.ru_utime-before.ru_utime,process_system_cpu_seconds=after.ru_stime-before.ru_stime,peak_rss_bytes=rss(),cpu_threads=4,gpu_processes=0))
