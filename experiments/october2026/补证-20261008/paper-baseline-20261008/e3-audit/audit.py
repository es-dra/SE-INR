"""Retrospective E3 record-level audit. No reconstruction, training, or GPU."""
import json, hashlib, time, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[2] / '服务器结果-20261007/claude-20261005-e3'
def read(name):
    return json.loads((OLD / name).read_text())
def rank(x):
    x = np.asarray(x)
    return np.array([np.count_nonzero(x < v) + (np.count_nonzero(x == v)+1)/2 for v in x])
def hashfile(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__ == '__main__':
    wall0, cpu0 = time.monotonic(), time.process_time()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    for name, sha in protocol['input_sha256'].items():
        assert hashfile(OLD/name) == sha, name
    assert hashfile(Path(__file__)) == protocol['audit_source_sha256']
    prior = {int(r['task'].split('_x')[1].split('_')[0]):r for r in read('d8_single.json') if r['task'].endswith('n0.002')}
    scales, bands, datasets = [2,3,8], ['b1','b2','b3'], ['Set14','BSD100','Urban100']
    nets = ['liif','lte','sc-inr-eq','sc-inr-nosinc']
    rng = np.random.default_rng(0)
    cells, counts, duplicate_counts = [], {}, {}
    def boot(records, func):
        v = np.array([func(r) for r in records])
        idx = rng.integers(0,len(v),(10000,len(v)))
        means = v[idx].mean(axis=1)
        return [round(float(v.mean()),4),round(float(np.percentile(means,2.5)),4),round(float(np.percentile(means,97.5)),4),len(v)]
    for s in scales:
        result = read(f'out/e3_x{s}.json')
        counts[s] = {ds:sum(r['set']==ds for r in result['per_image']) for ds in datasets}
        ids = [(r['set'],r['img']) for r in result['per_image']]
        duplicate_counts[s] = len(ids)-len(set(ids))
        for b in bands:
            rp = prior[s]['R_'+b]
            for ds in datasets:
                pool = result['pooled']; get = lambda k:pool[f'{ds}|{k}|{b}']
                nr = {n:get(n) for n in nets}
                records = [r for r in result['per_image'] if r['set']==ds]
                c = dict(scale=s,band=b,set=ds,prior=round(rp,4),zone='A' if rp>=.8 else ('C' if rp<.1 else 'B'),generic=get('lmmse_generic'),oracle=get('lmmse_oracle'),bicubic=get('bicubic'),nets=nr,net_range=max(nr.values())-min(nr.values()),netmax_minus_oracle=max(nr.values())-get('lmmse_oracle'),netmin_minus_oracle=min(nr.values())-get('lmmse_oracle'))
                c['ci_netmean_minus_oracle(per-image)'] = boot(records,lambda r:np.mean([r[f'{n}_{b}'] for n in nets])-r[f'lmmse_oracle_{b}'])
                c['ci_oracle_minus_generic(per-image)'] = boot(records,lambda r:r[f'lmmse_oracle_{b}']-r[f'lmmse_generic_{b}'])
                cells.append(c)
    original = read('e3_result.json')
    max_numeric_diff = 0.
    for a,b in zip(cells,original['cells']):
        for key in a:
            if isinstance(a[key],(float,int)):
                max_numeric_diff=max(max_numeric_diff,abs(a[key]-b[key]))
            else:
                assert a[key]==b[key], (key,a[key],b[key])
    def rho(x,y):
        return float(np.corrcoef(rank(x),rank(y))[0,1])
    r = rho([prior[c['scale']]['R_'+c['band']] for c in cells],[c['generic'] for c in cells])
    groups = {z:[c for c in cells if c['zone']==z] for z in 'ABC'}
    verdict = dict(spearman=r,C_n=len(groups['C']),C_pass_003=sum(c['netmin_minus_oracle']>0 and c['net_range']<=.03 for c in groups['C']),C_pass_002=sum(c['netmin_minus_oracle']>0 and c['net_range']<=.02 for c in groups['C']),A_n=len(groups['A']),A_pass=sum(c['netmax_minus_oracle']<=.05 for c in groups['A']),A_largest_net_oracle_gap=max(c['netmax_minus_oracle'] for c in groups['A']),C_smallest_net_oracle_gap=min(c['netmin_minus_oracle'] for c in groups['C']),C_largest_net_oracle_gap=max(c['netmax_minus_oracle'] for c in groups['C']),oracle_generic_gap={z:float(np.mean([c['oracle']-c['generic'] for c in g])) for z,g in groups.items()})
    verdict['generic_monotone_pairs'] = sum(all(v[i]>v[i+1] for i in range(2)) for v in [[next(c['generic'] for c in cells if c['scale']==s and c['band']==b and c['set']==ds) for s in scales] for b in bands for ds in datasets])
    payload=dict(status='record-level independent recalculation passed',input_sha256=protocol['input_sha256'],counts=counts,duplicate_counts=duplicate_counts,bootstrap_replicates=10000,bootstrap_seed=0,max_cell_numeric_difference=max_numeric_diff,verdicts=verdict,cells=cells,wall_seconds=time.monotonic()-wall0,cpu_seconds=time.process_time()-cpu0,scope='Saved scalar recovery and pooled records only; no pixel predictions, observation, periodogram, energy sums, CG residuals or checkpoints re-evaluated.')
    assert payload['cpu_seconds'] <= protocol['cpu_seconds_budget']
    assert time.time() <= protocol['absolute_deadline_unix']
    (ROOT/'record-recalculation.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('cells','input_sha256')},indent=2))
