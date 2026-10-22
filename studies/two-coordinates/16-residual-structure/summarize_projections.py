"""Summarize saved-output projection geometry without training or inference.

Output is a new immutable directory. Fits are retrospective and increasing fit
flexibility is not evidence for a causal explanation or out-of-sample prediction.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


MODELS = ["liif", "lte", "sc-inr"]
SEEDS, SCALES, MUS = [1, 2], [2, 3, 4, 6], [.5, 1.3333, 2.]
LEVELS = ["pooled", "A", "B", "C", "D"]
PAIRS = [(2, 3), (3, 4), (4, 6), (2, 4), (3, 6), (2, 6)]


def read_json(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def write_json(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return float(a/b) if b != 0 else None


def load(source):
    data = read_json(source)
    records = data["records"]
    images = sorted({row["image"] for row in records})
    if len(images) != 14 or len(records) != 1008:
        raise ValueError("Expected 14 images and 1008 complete records")
    names = data["layer_stat_names"]
    base_names = data["base_stat_names"]
    shape = (14, 3, 2, 4, 3, 18)
    base = np.full(shape+(len(base_names),), np.nan)
    values = {level:np.full(shape+(len(names),), np.nan) for level in LEVELS[1:]}
    quad = np.full(shape, np.nan)
    guards = {level:[] for level in LEVELS[1:]}
    seen = set()
    for row in records:
        key = (images.index(row["image"]), MODELS.index(row["model"]),
               SEEDS.index(row["seed"]), SCALES.index(row["scale"]), MUS.index(row["mu"]))
        if key in seen:
            raise ValueError("Duplicate record")
        seen.add(key)
        base[key] = np.vstack((row["base_bins"],row["base_dc"]))
        quad[key] = row["D_quadrature_energy_bins"]+[row["D_quadrature_energy_dc"]]
        for level in LEVELS[1:]:
            values[level][key] = np.vstack((row["layers"][level]["bins"],row["layers"][level]["dc"]))
            guards[level].append(row["layers"][level]["guard"])
    if not np.isfinite(base).all() or not np.isfinite(quad).all() or not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Incomplete/nonfinite source arrays")
    # Fit one w per model/scale/mu/bin after pooling images and seeds, as stage 15.
    mean = base.mean(axis=(0, 2))
    C,U,E,V,K,D,Z,G = [base[...,i] for i in range(8)]
    w = np.divide(mean[...,4],mean[...,1],out=np.zeros_like(mean[...,1]),where=mean[...,1]>0)
    w = w[None,:,None,:,:,:]
    pooled = np.zeros_like(values["A"])
    f = {name:i for i,name in enumerate(names)}
    eps = Z-2*w*K+w*w*U
    e_eps, p_eps = G-K-w*C, K-w*U
    scalar_energy, scalar_cross = (1-w)**2*U, 2*(1-w)*C
    residual_scalar_cross = -2*(1-w)*p_eps
    for name,array in {
        "epsilon_energy":eps,"e_epsilon":e_eps,"P_epsilon":p_eps,
        "scalar_change_energy":scalar_energy,"scalar_error_cross":scalar_cross,
        "scalar_cost":scalar_energy+scalar_cross,"epsilon_error_cross":-2*e_eps,
        "epsilon_scalar_cross":residual_scalar_cross,
        "H":eps-2*e_eps+residual_scalar_cross,"D":D,
        "change_energy":Z+U-2*K,"error_change_cross":-2*(G-K-C),
        "negative_w_native_energy":np.where(w<0,U,0),
        "negative_w_perturbed_energy":np.where(w<0,Z,0),
    }.items():
        pooled[...,f[name]]=array
    for name in ["zero_P_new_output_energy","zero_P_coefficient_count","small_P_native_energy","small_P_output_energy","small_P_coefficient_count"]:
        pooled[...,f[name]]=values["A"][...,f[name]]
    counts=np.asarray(data["metadata"]["bin_coefficient_counts"]+[1])*3
    pooled[...,f["negative_w_coefficient_count"]]=np.where(w<0,counts,0)
    values["pooled"]=pooled
    np.testing.assert_allclose(pooled[...,f["scalar_cost"]]+pooled[...,f["H"]],D,atol=2e-12,rtol=2e-10)
    np.testing.assert_allclose(quad+values["D"][...,f["zero_P_new_output_energy"]],values["D"][...,f["epsilon_energy"]],atol=2e-12,rtol=2e-10)
    # Pooled -> A is nested only after aggregating the population used to fit w.
    pooled_e=pooled[...,f["epsilon_energy"]].mean(axis=(0,2))
    a_e=values["A"][...,f["epsilon_energy"]].mean(axis=(0,2))
    if np.min(pooled_e-a_e)<-2e-12:
        raise ValueError("pooled->A residual energy unexpectedly increases")
    return data, images, base, values, quad, guards, f


def summarize(source,out):
    data,images,base,values,quad,guards,f=load(source)
    rows,model_rows,band_rows,image_rows,pair_rows=[],[],[],[],[]
    d_factor_rows=[]
    for scope,stop in [("r_lt_0.5",16),("complete_nonDC",17)]:
        # image,model,seed,scale,mu,field; fields are additive before averaging.
        summed={level:v[...,:stop,:].sum(axis=-2) for level,v in values.items()}
        averaged={level:v.mean(axis=(0,1,2)) for level,v in summed.items()}
        for mi,mu in enumerate(MUS):
            for si,scale in enumerate(SCALES):
                h=2*(mu-1)/scale
                z=float(base[...,:stop,6].sum(axis=-1).mean(axis=(0,1,2))[si,mi])
                for level in LEVELS:
                    a=averaged[level][si,mi]
                    e=float(a[f["epsilon_energy"]])
                    fields={name:float(a[index]) for name,index in f.items()}
                    rows.append(dict(scope=scope,mu=mu,scale=scale,level=level,h=h,
                        **fields,epsilon_over_pooled=ratio(e,averaged["pooled"][si,mi,f["epsilon_energy"]]),
                        epsilon_over_A=ratio(e,averaged["A"][si,mi,f["epsilon_energy"]]),
                        epsilon_per_h2=e/(h*h),H_per_h2=float(a[f["H"]])/(h*h),
                        perturbed_energy=z,epsilon_over_perturbed=ratio(e,z),
                        negative_w_perturbed_fraction=ratio(a[f["negative_w_perturbed_energy"]],z),
                        zero_P_perturbed_fraction=ratio(a[f["zero_P_new_output_energy"]],z),
                        small_P_perturbed_fraction=ratio(a[f["small_P_output_energy"]],z)))
                    for model_i,model in enumerate(MODELS):
                        for seed_i,seed in enumerate(SEEDS):
                            a_ms=summed[level][:,model_i,seed_i,si,mi].mean(axis=0)
                            ep=float(a_ms[f["epsilon_energy"]])
                            ep_pool=float(summed["pooled"][:,model_i,seed_i,si,mi,f["epsilon_energy"]].mean())
                            model_rows.append(dict(scope=scope,mu=mu,scale=scale,level=level,model=model,seed=seed,
                                epsilon_energy=ep,H=float(a_ms[f["H"]]),epsilon_error_cross=float(a_ms[f["epsilon_error_cross"]]),
                                epsilon_over_pooled=ratio(ep,ep_pool),epsilon_per_h2=ep/(h*h),H_per_h2=float(a_ms[f["H"]])/(h*h)))
            for start,end in PAIRS:
                i,j=SCALES.index(start),SCALES.index(end)
                delta_cost=averaged["pooled"][i,mi,f["D"]]-averaged["pooled"][j,mi,f["D"]]
                for level in LEVELS:
                    a,b=averaged[level][i,mi],averaged[level][j,mi]
                    pair_rows.append(dict(scope=scope,mu=mu,start=start,end=end,level=level,
                        delta_D=float(delta_cost),delta_epsilon=float(a[f["epsilon_energy"]]-b[f["epsilon_energy"]]),
                        delta_H=float(a[f["H"]]-b[f["H"]]),delta_scalar_cost=float(a[f["scalar_cost"]]-b[f["scalar_cost"]]),
                        delta_epsilon_error_cross=float(a[f["epsilon_error_cross"]]-b[f["epsilon_error_cross"]]),
                        H_share_of_cost_decrease=ratio(a[f["H"]]-b[f["H"]],delta_cost)))
        # D-energy=Z*eta. Primary decomposition fits eta to pooled energy per
        # model/bin, then averages model contributions; no factor extrapolation.
        z=base[...,:stop,6].mean(axis=(0,2)) # model,scale,mu,band
        q=values["D"][...,:stop,f["epsilon_energy"]].mean(axis=(0,2))
        eta=np.divide(q,z,out=np.zeros_like(q),where=z>0)
        if not np.all((eta>=-1e-10)&(eta<=1+1e-10)):
            raise ValueError("D residual fraction outside geometric [0,1] bound")
        for mi,mu in enumerate(MUS):
            for start,end in PAIRS:
                i,j=SCALES.index(start),SCALES.index(end)
                for normalized in [False,True]:
                    zi=z[:,i,mi].copy();zj=z[:,j,mi].copy()
                    if normalized:
                        zi/=(2*(mu-1)/start)**2;zj/=(2*(mu-1)/end)**2
                    etai,etaj=eta[:,i,mi],eta[:,j,mi]
                    amount=(zi-zj)*(etai+etaj)/2
                    fraction=(etai-etaj)*(zi+zj)/2
                    delta=zi*etai-zj*etaj
                    np.testing.assert_allclose(amount+fraction,delta,atol=2e-14,rtol=1e-11)
                    for model_i,model in list(enumerate(MODELS))+[(None,"mean")]:
                        select=lambda a:float(a.sum(axis=-1).mean() if model_i is None else a[model_i].sum())
                        da,df,dd=select(amount),select(fraction),select(delta)
                        d_factor_rows.append(dict(scope=scope,mu=mu,start=start,end=end,model=model,
                            normalization="per_h2" if normalized else "raw",delta_D_energy=dd,
                            output_energy_factor=da,residual_fraction_factor=df,
                            output_energy_factor_share=ratio(da,dd),residual_fraction_factor_share=ratio(df,dd)))
        # Frequency / image outputs are primary-scope only to avoid duplication.
        if scope!="r_lt_0.5":continue
        for mi,mu in enumerate(MUS):
            for si,scale in enumerate(SCALES):
                h=2*(mu-1)/scale
                for band in range(16):
                    for level in LEVELS:
                        a=values[level][:,:,:,si,mi,band].mean(axis=(0,1,2))
                        z_band=float(base[:,:,:,si,mi,band,6].mean())
                        band_rows.append(dict(mu=mu,scale=scale,level=level,band=band,r_low=band/32,r_high=(band+1)/32,
                            epsilon_energy=float(a[f["epsilon_energy"]]),H=float(a[f["H"]]),
                            epsilon_error_cross=float(a[f["epsilon_error_cross"]]),epsilon_per_h2=float(a[f["epsilon_energy"]])/(h*h),
                            perturbed_energy=z_band,epsilon_over_perturbed=ratio(a[f["epsilon_energy"]],z_band),
                            negative_w_perturbed_energy=float(a[f["negative_w_perturbed_energy"]]),
                            small_P_output_energy=float(a[f["small_P_output_energy"]])))
                for ii,image in enumerate(images):
                    for level in LEVELS:
                        a=summed[level][ii,:,:,si,mi].mean(axis=(0,1))
                        z_image=float(base[ii,:,:,si,mi,:16,6].sum(axis=-1).mean())
                        image_rows.append(dict(mu=mu,scale=scale,level=level,image=image,
                            epsilon_energy=float(a[f["epsilon_energy"]]),H=float(a[f["H"]]),
                            epsilon_error_cross=float(a[f["epsilon_error_cross"]]),epsilon_per_h2=float(a[f["epsilon_energy"]])/(h*h),
                            perturbed_energy=z_image,epsilon_over_perturbed=ratio(a[f["epsilon_energy"]],z_image)))
    # Quantify frequency variation and channel variation with both nested orders.
    # These are fit reductions, not physical causal shares.
    fit_rows=[]
    for row in rows:
        if row["scope"]!="r_lt_0.5" or row["level"]!="pooled":continue
        chosen={r["level"]:r for r in rows if r["scope"]==row["scope"] and r["mu"]==row["mu"] and r["scale"]==row["scale"]}
        e={k:v["epsilon_energy"] for k,v in chosen.items()}
        frequency=((e["A"]-e["C"])+(e["B"]-e["D"]))/2
        channel=((e["A"]-e["B"])+(e["C"]-e["D"]))/2
        np.testing.assert_allclose(frequency+channel+e["D"],e["A"],atol=2e-14,rtol=1e-10)
        fit_rows.append(dict(mu=row["mu"],scale=row["scale"],pooled_energy=e["pooled"],image_seed_fit_reduction=e["pooled"]-e["A"],
            A_energy=e["A"],A_to_B_reduction=e["A"]-e["B"],A_to_C_reduction=e["A"]-e["C"],
            B_to_D_reduction=e["B"]-e["D"],C_to_D_reduction=e["C"]-e["D"],
            frequency_fit_reduction_order_average=frequency,channel_fit_reduction_order_average=channel,
            D_energy=e["D"],frequency_share_of_A=ratio(frequency,e["A"]),channel_share_of_A=ratio(channel,e["A"]),D_share_of_A=ratio(e["D"],e["A"])))
    for name,table in [("scales",rows),("model_seed_scales",model_rows),("bands",band_rows),
                       ("images",image_rows),("pairs",pair_rows),("D_energy_factors",d_factor_rows),("fit_reductions",fit_rows)]:
        write_csv(out/f"{name}.csv",table)
    summary={
        "source":{"path":str(source.resolve()),"sha256":hashlib.sha256(source.read_bytes()).hexdigest()},
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"retrospective saved-output CPU analysis; no training, model loading or inference",
        "averaging":"14 images x 3 models x 2 seeds, equal weight; pooled w fitted separately per model/scale/mu/bin across images and seeds",
        "D_factor_decomposition":"epsilon_D=Z*eta per model/rbin; eta is an output-energy-weighted residual fraction, quadrature except exact-zero P. Symmetric factor decomposition is descriptive algebra, not independent causal effects.",
        "h_definition":"2*(mu-1)/scale, cell-input difference in feature-grid units; dividing finite response by h^2 is not an infinitesimal derivative",
        "fit_reduction_warning":"A->B and A->C are separate branches. Order-averaged frequency/channel reductions allocate overlap symmetrically but are not physical causal contributions.",
        "guards":{level:{"minimum_positive_raw_group_power":min(g["minimum_positive_raw_group_power"] for g in entries if g["minimum_positive_raw_group_power"] is not None),
                         "maximum_abs_gain":max(g["maximum_abs_gain"] for g in entries),
                         "zero_power_groups_total":sum(g["zero_power_groups"] for g in entries),
                         "small_positive_power_groups_total":sum(g["positive_power_groups_at_or_below_1e-24"] for g in entries),
                         "clipped_groups_total":sum(g["clipped_groups"] for g in entries)} for level,entries in guards.items()},
        "scales":rows,"fit_reductions":fit_rows,"pairs":pair_rows,"D_energy_factors":d_factor_rows,
    }
    # Compact rankings and model/seed ranges used in prose, all specified at mu=2.
    summary["mu2_primary"]={}
    for level in LEVELS:
        r2={r["image"]:r for r in image_rows if r["level"]==level and r["mu"]==2 and r["scale"]==2}
        r6={r["image"]:r for r in image_rows if r["level"]==level and r["mu"]==2 and r["scale"]==6}
        ranking=sorted([dict(image=image,delta_epsilon=r2[image]["epsilon_energy"]-r6[image]["epsilon_energy"],delta_H=r2[image]["H"]-r6[image]["H"]) for image in images],key=lambda r:r["delta_epsilon"],reverse=True)
        br2={r["band"]:r for r in band_rows if r["level"]==level and r["mu"]==2 and r["scale"]==2}
        br6={r["band"]:r for r in band_rows if r["level"]==level and r["mu"]==2 and r["scale"]==6}
        branking=sorted([dict(band=b,r_low=b/32,r_high=(b+1)/32,delta_epsilon=br2[b]["epsilon_energy"]-br6[b]["epsilon_energy"],delta_H=br2[b]["H"]-br6[b]["H"]) for b in range(16)],key=lambda r:r["delta_epsilon"],reverse=True)
        model_seed_trends=[]
        for model in MODELS:
            for seed in SEEDS:
                select=lambda scale:next(r for r in model_rows if r["scope"]=="r_lt_0.5" and r["mu"]==2 and r["level"]==level and r["model"]==model and r["seed"]==seed and r["scale"]==scale)
                a,b=select(2),select(6)
                model_seed_trends.append(dict(model=model,seed=seed,
                    epsilon_x2=a["epsilon_energy"],epsilon_x6=b["epsilon_energy"],
                    epsilon_x6_over_x2=ratio(b["epsilon_energy"],a["epsilon_energy"]),
                    epsilon_per_h2_x6_over_x2=ratio(b["epsilon_per_h2"],a["epsilon_per_h2"]),
                    H_x2=a["H"],H_x6=b["H"],delta_H=a["H"]-b["H"]))
        total_image=sum(r["delta_epsilon"] for r in ranking)
        total_band=sum(r["delta_epsilon"] for r in branking)
        summary["mu2_primary"][level]={"image_decrease_ranking":ranking,"band_decrease_ranking":branking,
            "top3_image_share_of_epsilon_decrease":ratio(sum(r["delta_epsilon"] for r in ranking[:3]),total_image),
            "top3_band_share_of_epsilon_decrease":ratio(sum(r["delta_epsilon"] for r in branking[:3]),total_band),
            "model_seed_trends":model_seed_trends}
    write_json(out/"summary.json",summary)
    plot(rows,d_factor_rows,out)
    write_json(out/"check.json",{"passed":True,"source_records":len(data["records"]),"all_arrays_finite":True,"pooled_exact_identity":True,"D_quadrature_identity":True,"D_factor_identity":True,"fit_reduction_identity":True,"models":MODELS,"seeds":SEEDS,"scales":SCALES,"mus":MUS})
    return summary


def plot(rows,factors,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":220})
    selected={(r["level"],r["scale"]):r for r in rows if r["scope"]=="r_lt_0.5" and r["mu"]==2}
    fig,axes=plt.subplots(1,3,figsize=(12.3,3.6),layout="constrained")
    colors={"pooled":"#5d6470","A":"#4a79a5","B":"#39a58c","C":"#d79936","D":"#b94c64"}
    for level in LEVELS[1:]:
        axes[0].plot(SCALES,[selected[level,s]["epsilon_over_pooled"] for s in SCALES],"o-",color=colors[level],label=level)
    axes[0].set(title="A. Remaining real-fit residual",xlabel="SR scale",ylabel="Residual energy / pooled residual",xticks=SCALES,ylim=(0,1))
    axes[0].legend(title="Fit",frameon=False,ncol=2)
    for level in LEVELS:
        axes[1].plot(SCALES,[selected[level,s]["epsilon_per_h2"]*1e5 for s in SCALES],"o-",color=colors[level],label=level)
    axes[1].set(title="B. Residual per finite input step²",xlabel="SR scale",ylabel="Residual energy / h² (×10⁻⁵)",xticks=SCALES)
    axes[1].legend(frameon=False,ncol=2,fontsize=8)
    split={r["model"]:r for r in factors if r["scope"]=="r_lt_0.5" and r["mu"]==2 and r["start"]==2 and r["end"]==6 and r["normalization"]=="raw"}
    labels=MODELS+["mean"]
    amount=np.array([split[m]["output_energy_factor_share"]*100 for m in labels])
    fraction=np.array([split[m]["residual_fraction_factor_share"]*100 for m in labels])
    positions=np.arange(len(labels))
    axes[2].bar(positions,amount,color="#4a79a5",label="Output energy Z")
    axes[2].bar(positions,fraction,bottom=amount,color="#d79936",label="Residual fraction η")
    axes[2].set(title="C. D decrease: fixed-band split",xlabel="×2 to ×6, symmetric allocation",ylabel="Share of residual-energy decrease (%)",xticks=positions,xticklabels=["LIIF","LTE","SC-INR","Mean"],ylim=(0,100))
    axes[2].legend(frameon=False,fontsize=8,loc="center left")
    fig.supxlabel("Saved Set14 crops, cell ×2; 0 < r < 0.5. A: image/bin, B: image/bin/channel, C: image/frequency, D: image/frequency/channel.",fontsize=8)
    fig.savefig(out/"projection-geometry.png",bbox_inches="tight")
    fig.savefig(out/"projection-geometry.pdf",bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    output=Path(args.out).resolve();output.mkdir(exist_ok=False)
    summary=summarize(Path(args.input).resolve(),output)
    for row in summary["scales"]:
        if row["scope"]=="r_lt_0.5" and row["mu"]==2:
            print(row["level"],row["scale"],"eps",row["epsilon_energy"],"H",row["H"],"eps/pool",row["epsilon_over_pooled"],"eps/h2",row["epsilon_per_h2"])
    print("PASS",output)


if __name__=="__main__":main()
