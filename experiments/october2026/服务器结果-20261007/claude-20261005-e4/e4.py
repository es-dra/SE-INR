# E4 summary per E4-汇总口径-20261005.md ; reads only result files, adds no new criteria
import json, numpy as np
from scipy.stats import spearmanr
W = "/workspace/SE-INR/"
e1 = json.load(open(W + "claude-20261004-e1/e1_result_v2.json"))
e3 = json.load(open(W + "claude-20261005-e3/e3_result.json"))
f1 = json.load(open(W + "claude-20261005-f1/f1_result.json"))
L = []
def row(exp, crit, thr, obs, v, src): L.append(f"| {exp} | {crit} | {thr} | {obs} | {v} | {src} |")
f = lambda x: "%.2f" % x
ci = lambda c: f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]"
V = e1["verdicts"]; s1 = "e1_result_v2.json"
row("E1", "P1 σ_n=0：低偏置最好 − ff 最优 σ", "≥ +1 dB", f"{V['P1']['0.0']['low']}−{V['P1']['0.0']['ff']} {ci(V['P1']['0.0']['diff'])}", V['P1']['0.0']['v'], s1)
row("E1", "P1 σ_n=0.005：同上", "≥ +1 dB", f"{V['P1']['0.005']['low']}−{V['P1']['0.005']['ff']} {ci(V['P1']['0.005']['diff'])}", V['P1']['0.005']['v'], s1)
row("E1", "**P1 总**", "两档都成立", "", V['P1']['v'], s1)
row("E1", "P2a σ_n=0.02：ff 最优 − mrgrid", "≥ 3 dB", ci(V['P2']['a']['diff']), V['P2']['a']['v'], s1)
for k, b in V['P2']['b'].items(): row("E1", f"P2b σ_n=0.02：{k}", "CI ⊂ [−0.6, 0.6]", ci(b['diff']), b['v'], s1)
row("E1", "**P2 总**", "a 与 b 全成立", "", V['P2']['v'], s1)
row("E1", "P3 D(0.02) − D(0.005)，D = oracle − 幂律最优", "≥ 0", f"D: {V['P3']['D']['0.005']:.2f}→{V['P3']['D']['0.02']:.2f}；{ci(V['P3']['diff'])}", V['P3']['v'], s1)
row("E1", "P4 赢家切换点", "∈ [0.005, 0.02]", f"切换于 {V['P4']['first_switch']}", V['P4']['v'], s1)
s2 = "项目库 结果-E2-谱失配代价-20261003.md（原始文件在云端）"
row("E2", "H-E2a Spearman(后验预测, 实测)", "≥ 0.8", "0.90", "成立", s2)
row("E2", "H-E2b R<0.3：高估代价 > 低估代价", "≥ 80%", "4/4", "成立", s2)
row("E2", "H-E2b R>0.7：低估代价 > 高估代价", "≥ 80%", "6/7", "成立", s2)
row("E2", "H-E2c 中位 |预测 − 实测|", "≤ 0.05", "0.038", "成立", s2)
row("E2", "H-E2d Spearman(事前预测, 实测)", "≥ 0.7", "0.79", "成立", s2)
V3 = e3["verdicts"]; s3 = "e3_result.json"
row("E3", "H-E3a Spearman(事前 R_b, 实测 lmmse_generic)", "≥ 0.8", "%.3f" % V3['H-E3a']['spearman'], V3['H-E3a']['v'], s3)
row("E3", "H-E3b C 区：网络 > oracle 且极差 ≤ 0.03", "≥ 80% 格", f"{V3['H-E3b']['n_ok']}/{V3['H-E3b']['n']}", V3['H-E3b']['v'], s3)
row("E3", "H-E3c A 区：最好网络 − oracle ≤ 0.05", "≥ 80% 格", f"{V3['H-E3c']['n_ok']}/{V3['H-E3c']['n']}", V3['H-E3c']['v'], s3)
row("E3", "H-E3d lmmse_generic 随倍率单调下降", "9/9", f"{sum(m[3] for m in V3['H-E3d']['cells'])}/9", V3['H-E3d']['v'], s3)
mg = V3['H-E3e(探索)']['mean_gap']
row("E3", "H-E3e（探索）oracle − generic：B > A 且 B > C", "—", f"A {mg['A']:.3f} / B {mg['B']:.3f} / C {mg['C']:.3f}", V3['H-E3e(探索)']['v'], s3)
# F1 recomputed
g = lambda R: R * (1 - R) ** 2; sp = lambda a, b: float(spearmanr(a, b).correlation); s4 = "f1_result.json（脚本重算）"
F = {}
for pn in ("lte", "sc"):
    C = [c for c in f1[pn] if c["s"] in (2, 3, 4)]
    A = [c["L"] for c in C if c["Rlin"] >= 0.8]; Bz = [c["L"] for c in C if c["s"] == 3 and c["b"] == "b2"]
    F[pn] = dict(F1=sp([g(c["Rlin"]) for c in C], [c["L"] for c in C]), F2=sp([g(c["Rnet"]) for c in C], [c["L"] for c in C]), A=float(np.mean(A)), B=float(np.mean(Bz)), n=len(C))
F4 = sp([c["L"] for c in f1["lte"] if c["s"] in (2, 3, 4)], [c["L"] for c in f1["sc"] if c["s"] in (2, 3, 4)])
both = lambda k: "成立" if all(F[p][k] >= 0.5 for p in F) else "不成立"
row("F1", "H-F1 Spearman(g(R_lin), L)，两对各自", "≥ 0.5", f"LTE {F['lte']['F1']:.3f} / SC {F['sc']['F1']:.3f}", both("F1"), s4)
row("F1", "H-F2 Spearman(g(R_net), L)，两对各自", "≥ 0.5", f"LTE {F['lte']['F2']:.3f} / SC {F['sc']['F2']:.3f}", both("F2"), s4)
f3 = all(F[p]["A"] <= 0.01 and F[p]["B"] > F[p]["A"] for p in F)
row("F1", "H-F3 A 区平均 L ≤ 0.01 且 ×3 b2 L > A 区", "—", f"LTE {F['lte']['A']:.4f}, {F['lte']['B']:.4f}；SC {F['sc']['A']:.4f}, {F['sc']['B']:.4f}", "成立" if f3 else "不成立", s4)
row("F1", "H-F4 两对 L 的 Spearman", "≥ 0.5", f"{F4:.3f}", "成立" if F4 >= 0.5 else "不成立", s4)
print("## 表 1 预注册判据总表\n\n| 实验 | 判据 | 阈值 | 实测 | 结论 | 来源 |\n|---|---|---|---|---|---|"); print("\n".join(L))
# Table 2
T = []
win = V['P4']['win']; pred = {"0.0": "低偏置（P1）", "0.005": "低偏置（P1）", "0.01": "未预测", "0.02": "谱先验/正则（P2/P4）"}
for nz in ("0.0", "0.005", "0.01", "0.02"):
    w = win[nz]; grp = "低偏置" if w["group"] == "LOW" else "谱先验/正则"
    ok = "—" if pred[nz] == "未预测" else ("一致" if pred[nz].startswith(grp) else "不一致")
    T.append(f"| E1 多帧 ×4 T=16 σ_n={nz} | {pred[nz]} | {w['winner']}（{grp}），领先 {w['second']} {ci(w['gap'])} | {ok} |")
T.append("| E2 R<0.3 的（任务, 频带） | 高估代价 > 低估代价 | 4/4 | 一致 |")
T.append("| E2 R>0.7 的（任务, 频带） | 低估代价 > 高估代价 | 6/7（反例：稀疏 10% b1） | 6/7 一致 |")
cells = e3["cells"]
for z, lab in (("C", "网络 > lmmse_oracle"), ("A", "最好网络 − oracle ≤ 0.05"), ("B", "未预测")):
    cz = [c for c in cells if c["zone"] == z]
    if z == "C": n = sum(c["netmin_minus_oracle"] > 0 for c in cz); obs = f"{n}/{len(cz)} 格网络全部 > oracle；网络 − oracle 范围 {min(c['netmin_minus_oracle'] for c in cz):+.3f} … {max(c['netmax_minus_oracle'] for c in cz):+.3f}"
    elif z == "A": n = sum(c["netmax_minus_oracle"] <= 0.05 for c in cz); obs = f"{n}/{len(cz)}；最大 {max(c['netmax_minus_oracle'] for c in cz):+.3f}"
    else: n = None; obs = f"网络全部 > oracle：{sum(c['netmin_minus_oracle'] > 0 for c in cz)}/{len(cz)}；网络 − oracle {min(c['netmin_minus_oracle'] for c in cz):+.3f} … {max(c['netmax_minus_oracle'] for c in cz):+.3f}"
    T.append(f"| E3 单图 ×2/×3/×8 {z} 区（{len(cz)} 格） | {lab} | {obs} | {'—' if n is None else ('一致' if n == len(cz) else f'{n}/{len(cz)}')} |")
for pn, nm in (("lte", "LTE"), ("sc", "SC-INR")):
    T.append(f"| F1 {nm} 读出算子，A 区 | L ≈ 0（≤ 0.01） | 平均 L = {F[pn]['A']:.4f}；×3 b2（B 区）L = {F[pn]['B']:.4f} | 一致 |")
print("\n## 表 2 预测赢家 vs 实测赢家\n\n| 设置 | 预注册预测 | 实测 | 是否一致 |\n|---|---|---|---|"); print("\n".join(T))
# E3 sensitivity at 0.02
C = [c for c in cells if c["zone"] == "C"]
ok02 = [c for c in C if c["netmin_minus_oracle"] > 0 and c["net_range"] <= 0.02]
print(f"\n## E3 敏感性：极差阈值 0.02\n\nH-E3b 满足 {len(ok02)}/{len(C)} = {len(ok02)/len(C):.0%}（阈值 80%）→ {'成立' if len(ok02)/len(C) >= 0.8 else '不成立'}。")
print("不满足的格：" + "；".join(f"×{c['scale']} {c['band']} {c['set']} 极差 {c['net_range']:.3f}" for c in C if c not in ok02))
