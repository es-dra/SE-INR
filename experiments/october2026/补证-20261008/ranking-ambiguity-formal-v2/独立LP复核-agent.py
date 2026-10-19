"""Separate post-run LP audit, independent of experiment.py; stdout JSON only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,itertools,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
warnings.filterwarnings('ignore',category=OptimizeWarning)
root=Path(sys.argv[1]);started=time.monotonic();deadline=started+120
read=lambda n:json.loads((root/n).read_text())
P=read('protocol.json');selections=read('selections.json');exact=read('exact_diagnostics.json')
mult=np.array([2.,2.,2.,1.]);t=np.arange(8);Q=[]
for k in range(1,4):
 F=np.column_stack([np.cos(np.pi*k*t/4),np.sin(np.pi*k*t/4)])*.5;Q.append(F@F.T)
q=np.cos(np.pi*t)/np.sqrt(8);Q=np.array(Q+[np.outer(q,q)])
def shape(name):
 p=np.array([k**(-3. if name=='power3' else -2.) for k in range(1,5)])
 if name=='power2_swap_k1_k3':p=p[[2,1,0,3]]
 if name=='middle_spike':p=np.array([.1,3.,.1,.8])
 return p*8/(mult@p)
def setup(ph,nu,projected=False):
 A=np.eye(8)[[v for p in ph for v in [p,p+4]]];eye=np.eye(len(A));D=list(eye)
 for i,j in itertools.combinations(range(len(A)),2):D.extend([eye[i]+eye[j],eye[i]-eye[j]])
 D=np.array(D);M=np.array([[d@A@q@A.T@d for q in Q] for d in D]);W=np.linalg.pinv(A)@A if projected else np.eye(8);cf=[]
 for name in P['candidate_spectra']:
  C=sum(a*q for a,q in zip(shape(name),Q));S=A@C@A.T+nu*np.eye(len(A));B=C@A.T@np.linalg.inv(S);E=W@(np.eye(8)-B@A)
  cf.append((np.array([np.trace(E@q@E.T) for q in Q]),nu*np.sum((W@B)**2)))
 return M,cf
stats={k:dict(planned_endpoints=384 if k=='exact' else 11322,checked_endpoints=0,solver_failures=0,max_lower_violation=0.,max_upper_violation=0.,violations_over_1e_7=0,max_primal_residual=0.,examples=[]) for k in ['exact','selection']}
def check(record,M,lo,hi,cf,kind,label):
 st=stats[kind];Au=np.vstack([M,-M]);bu=np.concatenate([hi,-lo])
 for pair in record['pairs']:
  j,k=pair['j'],pair['k'];coeff=cf[j][0]-cf[k][0];offset=cf[j][1]-cf[k][1]
  for sign in [1,-1]:
   if time.monotonic()>=deadline:return False
   fit=linprog(sign*coeff,A_ub=Au,b_ub=bu,A_eq=mult[None],b_eq=[8],bounds=[(0,8)]*4,method='highs',options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
   st['checked_endpoints']+=1
   if not fit.success:
    st['solver_failures']+=1
    if len(st['examples'])<8:st['examples'].append({'id':label,'status':fit.status,'message':fit.message})
    continue
   opt=sign*float(fit.fun)+offset;violation=pair['lower']-opt if sign==1 else opt-pair['upper'];fld='max_lower_violation' if sign==1 else 'max_upper_violation';st[fld]=max(st[fld],violation)
   p=fit.x;res=max(0.,max(M@p-hi),max(lo-M@p),abs(mult@p-8),max(-p),max(p-8));st['max_primal_residual']=max(st['max_primal_residual'],res)
   if violation>1e-7:
    st['violations_over_1e_7']+=1
    if len(st['examples'])<8:st['examples'].append({'id':label,'pair':[j,k],'sign':sign,'opt':opt,'outer':[pair['lower'],pair['upper']],'violation':violation})
 return True
finished=True
for key,r in exact.items():
 ph,nu,name=json.loads(key)
 for projected in [False,True]:
  M,cf=setup(ph,nu,projected);v=M@shape(name)
  if not check(r['projected_intervals' if projected else 'full_intervals'],M,v-1e-10,v+1e-10,cf,'exact',key+str(projected)):finished=False;break
 if not finished:break
if finished:
 for r in selections:
  if r['intervals']['status']!='ok':continue
  M,cf=setup(r['phases'],r['noise_variance'])
  if not check(r['intervals'],M,np.array(r['direction_lower']),np.array(r['direction_upper']),cf,'selection',r['id']):finished=False;break
for st in stats.values():st['not_checked_endpoints']=st['planned_endpoints']-st['checked_endpoints']
print(json.dumps({'stage':'independent post-run CPU LP audit; frozen formal cohort unchanged','wall_budget_seconds':120,'elapsed_seconds':time.monotonic()-started,'complete':finished,'threads':1,'gpu_used':False,'stats':stats,'limits':'Independent floating LP solutions check conservative direction numerically; original dual multipliers absent and no interval arithmetic proof.'},ensure_ascii=False,indent=2))
