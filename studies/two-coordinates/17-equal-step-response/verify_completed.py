"""Post-completion independent verification; no inference and no change to frozen scoring."""
from pathlib import Path
import json,gzip,hashlib,collections
import numpy as np
root=Path("/workspace/SE-INR/runs/codex-20261023-equal-step-v1")
expected_frozen="a77f84a54fb09676a6e6703b345ff8ce30d42597bfc874f29561bff552f9fb06"
def sha(p):
 d=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):d.update(b)
 return d.hexdigest()
assert (root/"score/COMPLETE.json").exists()
assert sha(root/"FROZEN.sha256")==expected_frozen
for line in (root/"FROZEN.sha256").read_text().splitlines():
 digest,rel=line.split("  ",1)
 assert sha(root/rel)==digest,rel
protocol=json.loads((root/"protocol.json").read_text())
summary=json.loads((root/"score/summary.json").read_text())
complete=json.loads((root/"score/COMPLETE.json").read_text())
assert complete["summary_sha256"]==sha(root/"score/summary.json")
frequency_source=summary["detail_files"]["frequency.csv"]
assert sha(Path(frequency_source["path"]))==frequency_source["sha256"]
manifest=json.loads((root/"input_manifest.json").read_text())
datasets=list(protocol["datasets"])
scales,hs=protocol["scales"],protocol["steps"]
models,seeds=protocol["models"],protocol["seeds"]
allrows={}; errors={"D_identity":0.,"r_sum":0.,"u_sum":0.}
record_count=0
for model in models:
 for seed in seeds:
  folder=root/"outputs"/f"{model}-seed{seed}"
  receipt=json.loads((folder/"COMPLETE.json").read_text())
  assert receipt["records_sha256"]==sha(folder/"records.jsonl.gz")
  count=0
  with gzip.open(folder/"records.jsonl.gz","rt") as stream:
   for line in stream:
    r=json.loads(line); total=np.array(r["total"]); rb=np.array(r["r_bins"]); ub=np.array(r["u_bins"]); dc=np.array(r["dc"])
    key=(r["dataset"],r["image"],r["model"],r["seed"],r["scale"],r["h"])
    assert key not in allrows
    bands=np.stack([dc,rb[:6].sum(axis=0),rb[6:12].sum(axis=0),rb[12:].sum(axis=0)])
    allrows[key]=(total,bands)
    errors["D_identity"]=max(errors["D_identity"],abs(total[1]-total[0]+2*total[2]))
    errors["r_sum"]=max(errors["r_sum"],float(np.max(np.abs(rb.sum(axis=0)+dc-total))))
    errors["u_sum"]=max(errors["u_sum"],float(np.max(np.abs(ub.sum(axis=0)+dc-total))))
    count+=1
  assert count==receipt["records"]; record_count+=count
assert record_count==246*6*4*3==17712
assert max(errors.values())<2e-12
condition_results=[]; contrasts=[]; endpoint_results=[]; total_results=[]; frequency=[]
max_summary_delta=0.; max_ci_delta=0.
def interval(x):
 return [float(q) for q in np.quantile(x,[.025,.975])]
for di,ds in enumerate(datasets):
 images=[r["image"] for r in manifest["images"][ds]]
 # Independent direct indexing and averaging, rather than scorer multiplicity matrices.
 values=np.array([[[[allrows[(ds,img,mo,se,s,h)][0] for h in hs] for s in scales] for mo in models for se in seeds] for img in images])
 bands=np.array([[[[allrows[(ds,img,mo,se,s,h)][1] for h in hs] for s in scales] for mo in models for se in seeds] for img in images])
 values=values.mean(axis=1); bands=bands.mean(axis=1)
 point=values.mean(axis=0); bp=bands.mean(axis=0)
 indices=np.random.default_rng(20261023+di).integers(0,len(images),size=(10000,len(images)))
 assert hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()==summary["bootstrap"][ds]["indices_sha256"]
 for hi,h in enumerate(hs):
  # Pair whole images across every scale and statistic, retain fixed-checkpoint mean.
  draws=values[:, :, hi][indices].mean(axis=1)
  band_draws=bands[:, :, hi, :, :2][indices].mean(axis=1)
  j=point[:,hi,0]; bj=draws[:,:,0]
  ratios=j[1:]/j[:-1]; ratios_boot=bj[:,1:]/bj[:,:-1]
  low36=np.quantile(ratios_boot,.05/36,axis=0)
  low3=np.quantile(ratios_boot,.05/3,axis=0)
  high3=np.quantile(ratios_boot,1-.05/3,axis=0)
  for pi,(s0,s1) in enumerate(zip(scales[:-1],scales[1:])):
   orig=next(q for q in summary["adjacent_contrasts"] if (q["dataset"],q["h"],q["start_scale"],q["end_scale"])==(ds,h,s0,s1))
   fields={"J_ratio":float(ratios[pi]),"lower_bound_family_36":float(low36[pi]),"lower_bound_local_3":float(low3[pi]),"upper_bound_local_3":float(high3[pi])}
   for k,v in fields.items():max_summary_delta=max(max_summary_delta,abs(v-orig[k]))
   ci=interval(ratios_boot[:,pi]);max_ci_delta=max(max_ci_delta,abs(ci[0]-orig["J_ratio_ci_low"]),abs(ci[1]-orig["J_ratio_ci_high"]))
   contrasts.append({"dataset":ds,"h":h,"scales":[s0,s1],"ratio":fields["J_ratio"],"ci95":ci,"family_36_low":fields["lower_bound_family_36"],"primary_reversal":bool(low36[pi]>1.10)})
  local=bool(np.any(low3>1.10)); falling=bool(np.all(high3<.90))
  label="supported_material_reversal" if local else "supported_material_monotone_decrease" if falling else "unresolved_or_small"
  original=next(q for q in summary["conditions"] if (q["dataset"],q["h"])==(ds,h))
  assert original["within_condition_pass"]==local and original["diagnostic"]==label
  condition_results.append({"dataset":ds,"h":h,"local_pass":local,"diagnostic":label,"family_wide_reversal":bool(np.any(low36>1.10))})
  ep=float(j[-1]/j[0]); epci=interval(bj[:,-1]/bj[:,0])
  oe=next(q for q in summary["endpoint_contrasts"] if (q["dataset"],q["h"])==(ds,h))
  max_summary_delta=max(max_summary_delta,abs(ep-oe["J_ratio"]))
  max_ci_delta=max(max_ci_delta,abs(epci[0]-oe["J_ratio_ci_low"]),abs(epci[1]-oe["J_ratio_ci_high"]))
  endpoint={"dataset":ds,"h":h,"J_6_over_2":ep,"ci95":epci}
  for k,key in enumerate(["J","D","A","V","E"]):
   difference=float(point[-1,hi,k]-point[0,hi,k])
   dci=interval(draws[:,-1,k]-draws[:,0,k])
   max_summary_delta=max(max_summary_delta,abs(difference-oe[key+"_difference"]))
   max_ci_delta=max(max_ci_delta,abs(dci[0]-oe[key+"_difference_ci_low"]),abs(dci[1]-oe[key+"_difference_ci_high"]))
   endpoint[key+"_difference"]=difference
   endpoint[key+"_difference_ci95"]=dci
  endpoint_results.append(endpoint)
  for si,s in enumerate(scales):
   ot=next(q for q in summary["scale_totals"] if (q["dataset"],q["h"],q["scale"])==(ds,h,s))
   for k,key in enumerate(["J","D","A","V","E"]):
    max_summary_delta=max(max_summary_delta,abs(point[si,hi,k]-ot[key]))
    ci=interval(draws[:,si,k]);max_ci_delta=max(max_ci_delta,abs(ci[0]-ot[key+"_ci_low"]),abs(ci[1]-ot[key+"_ci_high"]))
   d_over_j=float(point[si,hi,1]/point[si,hi,0])
   d_over_j_ci=interval(draws[:,si,1]/draws[:,si,0])
   max_summary_delta=max(max_summary_delta,abs(d_over_j-ot["D_over_J"]))
   max_ci_delta=max(max_ci_delta,abs(d_over_j_ci[0]-ot["D_over_J_ci_low"]),abs(d_over_j_ci[1]-ot["D_over_J_ci_high"]))
   total_results.append({"dataset":ds,"h":h,"scale":s,"J":float(point[si,hi,0]),"D":float(point[si,hi,1]),"alignment_cost":float(-2*point[si,hi,2]),"J_ci95":interval(draws[:,si,0]),"D_ci95":interval(draws[:,si,1]),"D_over_J":d_over_j,"D_over_J_ci95":d_over_j_ci})
   frequency.append({"dataset":ds,"h":h,"scale":s,"band_names":["DC","0<r<0.1875","0.1875<=r<0.375","0.375<=r<=0.5"],"J_band_shares":(bp[si,hi,:,0]/point[si,hi,0]).tolist(),"J_band_share_ci95":[interval(band_draws[:,si,bi,0]/draws[:,si,0]) for bi in range(4)],"J_band_energies":bp[si,hi,:,0].tolist(),"J_band_energy_ci95":[interval(band_draws[:,si,bi,0]) for bi in range(4)],"D_band_contributions":bp[si,hi,:,1].tolist(),"D_band_contribution_ci95":[interval(band_draws[:,si,bi,1]) for bi in range(4)]})
primary=any(r["primary_reversal"] for r in contrasts)
strong=all(r["local_pass"] for r in condition_results)
direct=all(r["ci95"][0]>1 for r in endpoint_results if r["dataset"]=="Set14" and r["h"]>0)
preds={"P17-equal-step":"PASS" if primary else "FAIL","P17-all-conditions":"PASS" if strong else "FAIL","P17-Set14-positive":"PASS" if direct else "FAIL"}
for k,v in preds.items():assert summary["prediction_results"][k]["status"]==v
assert max_summary_delta<1e-10 and max_ci_delta<1e-10,(max_summary_delta,max_ci_delta)
output={"independent":True,"source":"all six original frozen-run records.jsonl.gz; no import of frozen scorer","run_root":str(root),"frozen_sha256":expected_frozen,"summary_sha256":sha(root/"score/summary.json"),"records":record_count,"images":246,"conditions":len(condition_results),"contrasts":len(contrasts),"checks":errors,"max_point_and_corrected_bound_difference":max_summary_delta,"max_95CI_difference":max_ci_delta,"predictions":preds,"condition_results":condition_results,"adjacent_contrasts":contrasts,"endpoint_results":endpoint_results,"totals":total_results,"frequency_source":frequency_source,"frequency":frequency}
print(json.dumps(output,separators=(",",":")))
