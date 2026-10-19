# 事后诊断，描述性：A3 贡献、A4 BSD100 抽样、A5 barbara 曲线、JPEG 块效应指标（只读 out/l1_*.json）
import json, numpy as np, random
from PIL import Image
from pathlib import Path
HR=240; K=np.arange(121); UE=[0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]; SIN=[2,3,4]; NETS=["liif","lte","sc-inr","lmmse_generic"]
def mb(s,i):
    u=2*s*K/HR; lo,hi=UE[i],UE[i+1]; return (u>=lo-1e-12)&((u<hi-1e-12) if i<6 else (u<=hi+1e-12))&(K>0)
def load(ds): return json.load(open(f"out/l1_inr_{ds}.json"))
def R(d,mn,s,i,imgs):
    m=mb(s,i); e=sum(np.asarray(d["E"][mn][str(s)][x])[m].sum() for x in imgs); t=sum(np.asarray(d["S"][x])[m].sum() for x in imgs); return 1-e/t
def Din(d,mn,imgs):
    r=np.array([[R(d,mn,s,i,imgs) for i in range(7)] for s in SIN]); return np.median(r.max(0)-r.min(0))
S14=load("Set14"); B=load("BSD100"); I14=sorted(S14["S"]); IB=sorted(B["S"])
print("A3 per-image share (%) of ΣS and ΣE[lte] in u-bins 4-5 (u 1-1.5, the bins driving D_in), at s=2 / s=4")
for x in I14:
    row=[]
    for s in (2,4):
        m=mb(s,3)|mb(s,4)
        St=sum(np.asarray(S14["S"][y])[m].sum() for y in I14); Et=sum(np.asarray(S14["E"]["lte"][str(s)][y])[m].sum() for y in I14)
        row.append(f"s{s}: S%={100*np.asarray(S14['S'][x])[m].sum()/St:5.1f} E%={100*np.asarray(S14['E']['lte'][str(s)][x])[m].sum()/Et:5.1f}")
    print(f"  {x:15s}", "  ".join(row))
print("A4 BSD100 random subsets n=14 (200 draws, seed 0): D_in median")
rng=random.Random(0); draws=[rng.sample(IB,14) for _ in range(200)]
for mn in NETS:
    v=np.array([Din(B,mn,s) for s in draws]); obs=Din(S14,mn,I14)
    print(f"  {mn:14s} Set14={obs:.4f} BSDn14 med={np.median(v):.4f} p95={np.percentile(v,95):.4f} max={v.max():.4f} P(>=Set14)={np.mean(v>=obs):.3f} P(>0.05)={np.mean(v>0.05):.3f}")
print("A5 barbara.png R(u) per scale (single image), u-bins 1..7")
for mn in NETS:
    for s in SIN: print(f"  {mn:14s} x{s}", np.round([R(S14,mn,s,i,["barbara.png"]) for i in range(7)],3).tolist())
print("A5 Set14 without barbara, R(u) bins 4-5 per scale (lte):", {s:np.round([R(S14,"lte",s,i,[y for y in I14 if y!="barbara.png"]) for i in (3,4)],3).tolist() for s in SIN})
print("A1 blockiness (mean |grad| on 8px boundaries / elsewhere, full image grey; ~1.0 = no 8x8 JPEG blocks)")
for fp in sorted(Path("/workspace/SE-INR/Data/Set14/HR").glob("*.png")):
    g=np.asarray(Image.open(fp).convert("L"),float); dx=np.abs(np.diff(g,axis=1)); dy=np.abs(np.diff(g,axis=0))
    bx=dx[:,7::8].mean()/np.delete(dx,np.s_[7::8],axis=1).mean(); by=dy[7::8].mean()/np.delete(dy,np.s_[7::8],axis=0).mean()
    print(f"  {fp.name:15s} {bx:.3f} {by:.3f}")
