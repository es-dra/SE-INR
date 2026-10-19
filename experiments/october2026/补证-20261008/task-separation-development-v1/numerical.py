"""Deterministic matrix audits of the frozen eight-point candidate; no fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='4'
import json, hashlib, time, resource, signal
from pathlib import Path
import numpy as np

START=time.monotonic();ABS=min(time.time()+120,1791442800.)
os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'task-separation-check-run-v1'
if ROOT.exists(): raise RuntimeError('New root required; no resume')
ROOT.mkdir()
def check():
    if time.time()>=ABS or time.monotonic()-START>=120: raise TimeoutError('Nonextending audit budget')
def stop(a,b): raise TimeoutError('Interrupted')
signal.signal(signal.SIGTERM,stop)
identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['protocol.json','numerical.py']}
(ROOT/'frozen.json').write_text(json.dumps(dict(identity=identity,started_unix=time.time(),deadline_unix=ABS,cpu_threads=4,gpu=0),indent=2)+'\n')
n=8;j=np.arange(n);Q=lambda k:2/n*np.cos(2*np.pi*k*(j[:,None]-j[None,:])/n)
D=Q(1)-Q(3);A1=np.eye(8)[[0,4,1,5]];A2=np.eye(8)[[0,4,2,6]]
rows=[];fail=[]
def H(C,A,nu): return np.linalg.solve(A@C@A.T+nu*np.eye(4),A@C).T
def risk(C,A,nu,B):
    E=np.eye(8)-B@A
    return np.trace(E@C@E.T+nu*B@B.T)
def close(v,w):
    if abs(v-w)>2e-12: raise AssertionError((v,w))
try:
    for theta in [.01,.1,.3,.5,.8,.99]:
        for nu in [.001,.01,.1,1.,10.,100.]:
            check();a=1+nu;Cp=np.eye(n)+theta*D;Cm=np.eye(n)-theta*D
            assert np.linalg.eigvalsh(Cp)[0]>0 and np.linalg.eigvalsh(Cm)[0]>0
            Sp=A1@Cp@A1.T+nu*np.eye(4);Sm=A1@Cm@A1.T+nu*np.eye(4)
            assert np.max(abs(A2@Cp@A2.T-A2@Cm@A2.T))<1e-14
            Hp1=H(Cp,A1,nu);Hm1=H(Cm,A1,nu);Hp2=H(Cp,A2,nu);Hm2=H(Cm,A2,nu)
            r1=risk(Cp,A1,nu,Hp1);r2=risk(Cp,A2,nu,Hp2)
            gap=theta**2*(1+2*nu-theta**2)/(a*(a*a-theta**2/2))
            gamma=8*a*theta**2*(a*a-a+.5)/(a*a-theta**2/2)**2
            wrong2=risk(Cp,A2,nu,Hm2)-r2
            midpoint=(Hp2+Hm2)/2
            mix=risk(Cp,A2,nu,midpoint)-r2
            close(r1-r2,gap);close(r2,8-(4+2*theta**2)/a)
            close(wrong2,8*theta**2/a);close(mix,2*theta**2/a)
            close(risk(Cm,A2,nu,midpoint),risk(Cp,A2,nu,midpoint))
            close(risk(Cp,A1,nu,Hm1)-r1,gamma)
            close(risk(Cm,A1,nu,Hp1)-risk(Cm,A1,nu,Hm1),gamma)
            rho=np.exp(.25*np.linalg.slogdet(Sp)[1]+.25*np.linalg.slogdet(Sm)[1]-.5*np.linalg.slogdet((Sp+Sm)/2)[1])
            close(rho,np.sqrt(1-theta**2/(2*a*a)))
            margin=2*theta**2/a-gap
            assert gap>0 and margin>0
            # Exponential sufficient K, not an exact optimum or practical cost claim.
            K=int(np.ceil(4*a*a/theta**2*np.log(gamma/(2*margin))))+1
            upper=.5*gamma*np.exp(-K*theta**2/(4*a*a))
            assert upper<margin
            rows.append(dict(theta=theta,nu=nu,oracle01=r1,oracle02=r2,gap=gap,
                wrong_regret02=wrong2,unrestricted_minimax_excess02=mix,
                wrong_regret01=gamma,bhattacharyya=rho,sufficient_K=K,
                selector01_upper_total=r1+upper,unrestricted02_total=8-4/a,
                numeric_gap_residual=abs(r1-r2-gap)))
            (ROOT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert len(rows)==36
    (ROOT/'COMPLETE').write_text('Deterministic identities only; not proof or novelty.\n')
except Exception as exc:
    fail.append(dict(type=type(exc).__name__,message=str(exc)))
finally:
    result=dict(n_cases=len(rows),failures=fail,full_wall_seconds=time.monotonic()-START,
        process_user_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        process_system_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        deadline_unix=ABS,cpu_threads=4,gpu=0,scope='matrix identities; no Monte Carlo, neural training or statistical cohort')
    (ROOT/'resources.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if fail: raise RuntimeError(fail)
