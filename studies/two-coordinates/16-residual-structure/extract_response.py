"""Read original frozen frequency rows, never execute a model."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(exist_ok=False)
    t=time.perf_counter()
    raw=a.rows.read_bytes()
    rows=json.loads(raw)
    accum=defaultdict(lambda: [0.0]*6)
    maxerr=0
    count=0
    for r in rows:
        for mu in ['0.5','1.3333','2.0']:
            v=r['err']['1.0']; d=r['err'][mu]-v; j=r['b'][mu]; cross=r['a'][mu]
            maxerr=max(maxerr,abs(d-j+2*cross))
            count+=1
            region='u<1' if r['hi']<=1+1e-9 else 'u>=1'
            for scope in ['all_nonDC',region]:
                key=(r['ds'],r['img'],r['model'],r['seed'],r['s'],float(mu),scope)
                vals=[v,r['tgt'],d,j,cross,1]
                for k,z in enumerate(vals):accum[key][k]+=z
    assert maxerr < 1e-10,maxerr
    result=[]
    for key,vals in sorted(accum.items()):
        result.append(dict(zip(['dataset','image','model','seed','scale','mu','scope'],key),V=vals[0],E=vals[1],D=vals[2],J=vals[3],cross=vals[4],bins=int(vals[5])))
    out={'source_path':str(a.rows),'source_sha256':hashlib.sha256(raw).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'rows':len(rows),'checked_cases':count,'identity_max_abs':maxerr,
         'units':'RGB mean MSE, inherited original /3/N^4 normalization; no additional division',
         'records':result,'seconds':time.perf_counter()-t,'model_calls':0}
    (a.out/'response_stats.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='records'}))


if __name__=='__main__':main()
