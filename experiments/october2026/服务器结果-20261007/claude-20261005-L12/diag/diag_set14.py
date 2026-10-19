# 事后诊断，描述性：Set14 D_in 偏大来源（只读 out/l1_*.json，不改判定）
import json, numpy as np, random
from PIL import Image
HR=240; K=np.arange(121); UE=[0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]; SIN=[2,3,4]; NETS=["liif","lte","sc-inr","lmmse_generic"]
def mb(s,i):
    u=2*s*K/HR; lo,hi=UE[i],UE[i+1]; return (u>=lo-1e-12)&((u<hi-1e-12) if i<6 else (u<=hi+1e-12))&(K>0)
def load(ds): return json.load(open(f"out/l1_inr_{ds}.json"))
def R(d,mn,s,i,imgs):
    m=mb(s,i); e=sum(np.asarray(d["E"][mn][str(s)][x])[m].sum() for x in imgs); t=sum(np.asarray(d["S"][x])[m].sum() for x in imgs); return 1-e/t
def Din(d,mn,imgs,ret=False):
    r=np.array([[R(d,mn,s,i,imgs) for i in range(7)] for s in SIN]); din=r.max(0)-r.min(0); return (np.median(din),din,r) if ret else np.median(din)
S14=load("Set14"); B=load("BSD100"); I14=sorted(S14["S"]); IB=sorted(B["S"])
print("A2 per-u-bin D_in on Set14 (bins: .25-.5 .5-.75 .75-1 1-1.25 1.25-1.5 1.5-1.75 1.75-2)")
for mn in NETS:
    med,din,r=Din(S14,mn,I14,True); print(f"  {mn:14s} med={med:.4f} bins=",np.round(din,3).tolist(), " argmax-scale per bin:",[SIN[j] for j in r.argmax(0)], " argmin:",[SIN[j] for j in r.argmin(0)])
print("A3 leave-one-out D_in median (Set14)")
for mn in NETS:
    base=Din(S14,mn,I14); loo={x:Din(S14,mn,[y for y in I14 if y!=x]) for x in I14}
    top=sorted(loo.items(),key=lambda kv:kv[1])[:3]; print(f"  {mn:14s} base={base:.4f} lowest-after-drop:",[(k,round(v,4)) for k,v in top], " max:",round(max(loo.values()),4))
print("A3 share of ΣS and ΣE(lte, x3) within u in [0.25,2] at s=3, per image")
m=np.zeros(121,bool)
for i in range(7): m|=mb(3,i)
St=sum(np.asarray(S14["S"][x])[m].sum() for x in I14); Et=sum(np.asarray(S14["E"]["lte"]["3"][x])[m].sum() for x in I14)
for x in sorted(I14,key=lambda x:-np.asarray(S14["S"][x])[m].sum()):
    print(f"  {x:15s} S%={100*np.asarray(S14[S][x])[m].sum()/St:5.1f} E%={100*np.asarray(S14[E][lte][3][x])[m].sum()/Et:5.1f}")
print("A4 BSD100 random subsets of n=14 (200 draws, seed 0), D_in median distribution")
rng=random.Random(0); draws=[rng.sample(IB,14) for _ in range(200)]
for mn in NETS:
    v=np.array([Din(B,mn,s) for s in draws]); obs=Din(S14,mn,I14)
    print(f"  {mn:14s} Set14={obs:.4f} BSD n14: median={np.median(v):.4f} p95={np.percentile(v,95):.4f} max={v.max():.4f} frac>=Set14={np.mean(v>=obs):.3f} frac>0.05={np.mean(v>0.05):.3f}")
json.dump({},open("diag/.placeholder","w"))
