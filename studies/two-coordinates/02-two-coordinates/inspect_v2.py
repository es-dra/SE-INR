import json,collections,numpy as np
rows=json.load(open("formal/rows.json"))
A=collections.defaultdict(collections.Counter);T=collections.Counter()
for r in rows:
  k=(r["model"],r["seed"],r["s"],r["lo"]);A[k].update(r["err"]);T[k]+=r["tgt"]
M=["liif","lte","sc-inr"]
for b in np.arange(0.5,2.0,0.25):
  print(f"u={b:.2f} "+" ".join(f"x{s}:{np.mean([1-A[(m,sd,s,b)]['1.0']/T[(m,sd,s,b)] for m in M for sd in [1,2,3]]):.3f}" for s in (2,3,4,6) if b<s))
for s in (2,3,4,6):
  g=[]
  for m in M:
    for sd in [1,2,3]:
      tot={mu:sum(A[k][mu] for k in A if k[:3]==(m,sd,s)) for mu in A[(m,sd,s,0.0)]}
      best=min(tot,key=tot.get); g.append(10*np.log10(tot["1.0"]/tot[best]))
  print(f"x{s}: gain of best fixed cell over native: mean {np.mean(g):.4f} dB max {max(g):.4f}")
