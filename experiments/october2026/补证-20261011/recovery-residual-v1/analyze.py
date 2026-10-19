"""Bounded retrospective decomposition of scale-recovery aggregation.

No models, fitting, image removal, new inference or training. The identities
are classical weighted-mean algebra; differences are not causal mechanisms.
"""
import datetime, hashlib, itertools, json, pathlib, resource, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def divergence(curve): return float(np.median(np.ptp(curve, axis=0)))
def describe(values):
    values=np.asarray(values)
    return dict(min=float(values.min()), q25=float(np.percentile(values,25)),
                median=float(np.median(values)), q75=float(np.percentile(values,75)),
                q90=float(np.percentile(values,90)), max=float(values.max()))

def main():
    start=time.perf_counter(); cpu=time.process_time()
    now=datetime.datetime.now(datetime.timezone.utc)
    protocol=read(ROOT/'protocol.json'); freeze=read(ROOT/'freeze.json')
    assert now.timestamp()<protocol['absolute_deadline_unix']
    assert sha(ROOT/'analyze.py')==freeze['source_sha256']
    assert sha(ROOT/'protocol.json')==freeze['protocol_sha256']
    for x in freeze['inputs']: assert sha(pathlib.Path(x['path']))==x['sha256']
    original=read(pathlib.Path(protocol['previous_robustness_results']))
    prior={(r['dataset'],r['model'],r['seed'],r['pooling']):r for r in original['l1']['rows']
           if r['domain']=='all' and r['scales']==[2,3,4]}
    edges=np.asarray(protocol['u_edges']); K=np.arange(121); scales=protocol['scales']
    groups=[]; geometry=[]; pairs=[]; max_identity=0.; max_primary=0.
    for ds in protocol['datasets']:
        data=read(pathlib.Path(protocol['l1_root'])/f'l1_inr_{ds}.json')
        names=sorted(data['S']); N=len(names)
        source=np.asarray([data['S'][im] for im in names],dtype=np.float64)
        masks={}
        for axis in ['u','absolute']:
            for s in scales:
                x=2*s*K/240 if axis=='u' else 6*K/240
                masks[axis,s]=[(x>=edges[b]-1e-12)&((x<edges[b+1]-1e-12) if b<6 else (x<=edges[b+1]+1e-12))&(K>0) for b in range(7)]
        S=np.asarray([[source[:,mask].sum(1) for mask in masks['u',s]] for s in scales]).transpose(0,2,1)
        assert S.shape==(3,N,7) and np.isfinite(S).all() and (S>0).all()
        w=S/S.sum(1,keepdims=True); fixed=w.mean(0)
        boot=np.random.default_rng(protocol['bootstrap_seed']).integers(0,N,size=(protocol['bootstrap_replicates'],N))
        gb=[]
        for si,s in enumerate(scales):
            for b in range(7):
                order=np.argsort(-w[si,:,b],kind='stable')
                gb.append(dict(scale=s,bin=b,u_bounds=edges[b:b+2].tolist(),
                    effective_images=float(1/np.sum(w[si,:,b]**2)),
                    energy_min=float(S[si,:,b].min()),energy_median=float(np.median(S[si,:,b])),
                    top1=dict(image=names[order[0]],weight=float(w[si,order[0],b])),
                    top3_weight=float(w[si,order[:3],b].sum()),
                    energy=S[si,:,b].tolist(),weights=w[si,:,b].tolist()))
        geometry.append(dict(dataset=ds,images=names,bins=gb))
        for model in protocol['models']:
          for seed in protocol['seeds']:
            key=model if seed==1 else f'{model}@s{seed}'
            E=np.asarray([[np.asarray([data['E'][key][str(s)][im] for im in names])[:,mask].sum(1)
                           for mask in masks['u',s]] for s in scales]).transpose(0,2,1)
            assert E.shape==S.shape and np.isfinite(E).all()
            R=1-E/S; pooled=1-E.sum(1)/S.sum(1); equal=R.mean(1); fixedcurve=np.sum(fixed[None,:,:]*R,axis=1)
            covariance=np.mean((S-S.mean(1,keepdims=True))*(R-R.mean(1,keepdims=True)),axis=1)/S.mean(1)
            error=float(np.max(np.abs(pooled-equal-covariance))); max_identity=max(max_identity,error)
            absolute=[]
            for s in scales:
                ee=np.asarray([data['E'][key][str(s)][im] for im in names])
                se=np.asarray([source[:,mask].sum(1) for mask in masks['absolute',s]]).T
                en=np.asarray([ee[:,mask].sum(1) for mask in masks['absolute',s]]).T
                absolute.append(1-en/se)
            absolute=np.asarray(absolute)
            individual=np.median(np.ptp(R,axis=0),axis=1)
            individual_absolute=np.median(np.ptp(absolute,axis=0),axis=1)
            Ds={'original_energy_pool':divergence(pooled),'equal_image':divergence(equal),
                'fixed_cross_scale_energy_weights':divergence(fixedcurve)}
            max_primary=max(max_primary,abs(Ds['original_energy_pool']-prior[ds,model,seed,'energy_pool']['D_u']),
                            abs(Ds['equal_image']-prior[ds,model,seed,'uniform_image_mean']['D_u']))
            perimage=[]; loo=[]
            for ii,im in enumerate(names):
                perimage.append(dict(image=im,D_u=float(individual[ii]),D_absolute=float(individual_absolute[ii]),
                    normalization_improves=bool(individual[ii]<individual_absolute[ii]),
                    strict_005=bool(individual[ii]<=.05),mean_energy_weight=float(fixed[ii].mean()),
                    max_energy_weight=float(w[:,ii,:].max()),R=R[:,ii,:].tolist()))
                keep=np.arange(N)!=ii
                lp=1-E[:,keep,:].sum(1)/S[:,keep,:].sum(1)
                le=R[:,keep,:].mean(1)
                loo.append(dict(image_removed=im,D_original=divergence(lp),D_equal=divergence(le),
                    delta_D_original=divergence(lp)-Ds['original_energy_pool']))
            current=[]
            for p,q in itertools.combinations(range(3),2):
              for b in range(7):
                within=.5*(w[p,:,b]+w[q,:,b])*(R[p,:,b]-R[q,:,b])
                migration=(w[p,:,b]-w[q,:,b])*.5*(R[p,:,b]+R[q,:,b])
                covgap=covariance[p,b]-covariance[q,b]
                gap=pooled[p,b]-pooled[q,b]
                er=max(abs(gap-within.sum()-migration.sum()),abs(gap-(equal[p,b]-equal[q,b])-covgap))
                max_identity=max(max_identity,float(er))
                row=dict(dataset=ds,model=model,seed=seed,scale_pair=[scales[p],scales[q]],bin=b,
                    pooled_gap=float(gap),equal_gap=float(equal[p,b]-equal[q,b]),covariance_gap=float(covgap),
                    within_image_component=float(within.sum()),weight_migration_component=float(migration.sum()),
                    mean_abs_individual_gap=float(np.mean(np.abs(R[p,:,b]-R[q,:,b]))),
                    mean_signed_individual_gap=float(np.mean(R[p,:,b]-R[q,:,b])),
                    positive_image_gaps=int(np.sum(R[p,:,b]-R[q,:,b]>0)),negative_image_gaps=int(np.sum(R[p,:,b]-R[q,:,b]<0)),
                    image_components=[dict(image=im,within=float(within[ii]),weight_migration=float(migration[ii])) for ii,im in enumerate(names)])
                current.append(row); pairs.append(row)
            # Paired image-cluster resampling; no new independent images or seed replicas.
            sb=S[:,boot,:].sum(2); eb=E[:,boot,:].sum(2)
            pb=1-eb/sb; rb=R[:,boot,:].mean(2)
            dp=np.median(np.ptp(pb,axis=0),axis=1); de=np.median(np.ptp(rb,axis=0),axis=1)
            groups.append(dict(dataset=ds,model=model,seed=seed,images=N,D=Ds,
                curves=dict(pooled=pooled.tolist(),equal=equal.tolist(),fixed=fixedcurve.tolist(),covariance=covariance.tolist()),
                per_image_summary=describe(individual),per_image_normalization_improves=int(np.sum(individual<individual_absolute)),
                per_image_strict_005=int(np.sum(individual<=.05)),per_image=perimage,
                leave_one_out=loo,leave_one_out_D_range=describe([r['D_original'] for r in loo]),
                largest_deletion_effect=sorted(loo,key=lambda r:abs(r['delta_D_original']),reverse=True)[:3],
                bootstrap=dict(replicates=len(boot),pool_D_interval=np.percentile(dp,[2.5,97.5]).tolist(),
                    equal_D_interval=np.percentile(de,[2.5,97.5]).tolist(),pool_minus_equal_D_interval=np.percentile(dp-de,[2.5,97.5]).tolist()),
                signed_gap_totals=dict(sum_abs_within=float(sum(abs(r['within_image_component']) for r in current)),
                    sum_abs_migration=float(sum(abs(r['weight_migration_component']) for r in current)),
                    sum_abs_pooled=float(sum(abs(r['pooled_gap']) for r in current)))))
    assert len(groups)==27 and len(pairs)==567
    assert max_primary<1e-12 and max_identity<1e-12
    summary=[]
    for ds in protocol['datasets']:
        rr=[r for r in groups if r['dataset']==ds]
        summary.append(dict(dataset=ds,groups=len(rr),
            original_strict_pass=sum(r['D']['original_energy_pool']<=.05 for r in rr),
            equal_strict_pass=sum(r['D']['equal_image']<=.05 for r in rr),
            fixed_strict_pass=sum(r['D']['fixed_cross_scale_energy_weights']<=.05 for r in rr),
            fixed_D_below_original=sum(r['D']['fixed_cross_scale_energy_weights']<r['D']['original_energy_pool'] for r in rr),
            median_image_D=describe([r['per_image_summary']['median'] for r in rr]),
            individual_strict_fraction=describe([r['per_image_strict_005']/r['images'] for r in rr]),
            individual_normalization_improvement_fraction=describe([r['per_image_normalization_improves']/r['images'] for r in rr])))
    cost=dict(start_utc=now.isoformat(),end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CPU_threads=1,nice=10,GPU=0,inference=0,training=0)
    assert cost['wall_seconds']<protocol['cumulative_cpu_analysis_budget_seconds']
    assert datetime.datetime.now(datetime.timezone.utc).timestamp()<protocol['absolute_deadline_unix']
    result=dict(scope=protocol['scope'],summary=summary,groups=groups,weight_geometry=geometry,pairs=pairs,
        verification=dict(max_original_reproduction_error=max_primary,max_identity_error=max_identity),cost=cost)
    save(ROOT/'results.json',result)
    save(ROOT/'output_manifest.json',dict(results_sha256=sha(ROOT/'results.json'),source_sha256=sha(ROOT/'analyze.py'),
        protocol_sha256=sha(ROOT/'protocol.json'),freeze_sha256=sha(ROOT/'freeze.json')))
    print(json.dumps(dict(summary=summary,verification=result['verification'],cost=cost),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
