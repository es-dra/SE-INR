"""Render verified saved results for the weekly report; no experiments or fitting."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

p = Path(__file__).resolve().parent
font_manager.fontManager.addfont(str(p / 'STHeiti Light.ttc'))
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(p / 'STHeiti Light.ttc')).get_name(),
    'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.unicode_minus': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
a = json.loads((p / 'robustness-input.json').read_text())
n = json.loads((p / 'negative-input.json').read_text())
blue, teal, red, grey = '#496780', '#168579', '#CB635B', '#6B7280'
rows = [r for r in a['l1']['rows'] if r['seed'] == 1 and r['pooling'] == 'energy_pool'
        and r['domain'] == 'all' and r['scales'] == [2, 3, 4]]
fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=True)
fig.subplots_adjust(left=.07, right=.98, bottom=.30, top=.76, wspace=.15)
fig.text(.06, .94, '线一：按倍率统一频率后，跨倍率差异下降', fontsize=22, weight='bold', color='#1C2D3C')
fig.text(.06, .875, '本周完成 3 模型 × 3 数据集 × 3 种子复核；27/27 组改善，降幅 80.0%–97.4%', fontsize=15, color=grey)
for ax, ds in zip(axes, ['Set14', 'BSD100', 'Urban100']):
    rr = [next(r for r in rows if r['dataset'] == ds and r['model'] == m) for m in ['liif', 'lte', 'sc-inr']]
    x = np.arange(3)
    for dx, key, color, label in [(-.18, 'D_nu', blue, '原始频率'), (.18, 'D_u', teal, '倍率归一化')]:
        vals = [r[key] for r in rr]
        bars = ax.bar(x + dx, vals, .34, color=color, label=label, zorder=3)
        for b, val in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, val+.009, f'{val:.3f}', ha='center', fontsize=11)
    ax.axhline(.05, color=red, ls='--', lw=1.2)
    ax.set_title(ds, fontsize=18, pad=18)
    ax.set_xticks(x, ['LIIF', 'LTE', 'SC-INR'])
    ax.set_ylim(0, .52)
    ax.set_yticks([0, .1, .2, .3, .4, .5])
    ax.grid(axis='y', alpha=.15, zorder=0)
    reduced = [100*(1-r['D_u']/r['D_nu']) for r in rr]
    ax.set_xlabel(f'差异下降 {min(reduced):.1f}%–{max(reduced):.1f}%', labelpad=14, color=teal)
axes[0].set_ylabel('跨倍率差异指标 D（越小越一致）')
axes[0].legend(frameon=False, fontsize=11, loc='upper left')
fig.text(.07, .17, '图示为原统计口径的种子1结果。虚线：预定一致性阈值 D = 0.05。', fontsize=12, color=grey)
fig.text(.07, .105, '未解决：27组中8组未达阈值，均来自Set14；改成每图等权后27/27通过，结论仍受统计权重影响。', fontsize=12, color=red)
fig.text(.07, .06, 'D为各频段恢复分数在×2/×3/×4间极差的中位数；两种频率坐标的频段位置不同，降幅不是PSNR提升。', fontsize=11, color=grey)
fig.savefig(p / 'frequency-recovery.png', dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={'width_ratios': [1, 1.25]})
fig.subplots_adjust(left=.07, right=.98, bottom=.30, top=.73, wspace=.35)
fig.text(.06, .94, '线二：负收益确实存在，现有预测尚未超过简单基线', fontsize=22, weight='bold', color='#1C2D3C')
fig.text(.06, .875, '32张图校准，12张图评价 × 2倍率；固定已有模型权重，比较原读出与零尺度读出', fontsize=14, color=grey)
ax = axes[0]
x = np.arange(2)
for dx, key, denom, color, label in [(-.18, 'band_metrics', 72, blue, '分频带：72项'), (.18, 'RGB_policies', 24, red, '整体RGB：24项')]:
    counts = [n['models'][m][key]['always_enable']['negative'] for m in ['sc-inr', 'lte']]
    vals = [100*c/denom for c in counts]
    bars = ax.bar(x+dx, vals, .34, color=color, label=label, zorder=3)
    for b, c, val in zip(bars, counts, vals):
        ax.text(b.get_x()+b.get_width()/2, val+1, f'{c}/{denom}', ha='center', fontsize=14)
ax.set_xticks(x, ['SC-INR', 'LTE'])
ax.set_ylim(0, 30)
ax.set_ylabel('负收益比例（%）')
ax.set_title('开启读出修正后，哪些结果变差？', fontsize=15, pad=17)
ax.grid(axis='y', alpha=.15, zorder=0)
ax.legend(frameon=False, fontsize=11, loc='upper right')
ax = axes[1]
labels = ['幅度/频谱基线', '基线＋修正方向', '双三次代理预测']
keys = ['norm_ridge', 'directional_ridge', 'proxy']
ys = np.arange(3)
bm = n['models']['sc-inr']['band_metrics']
detected = [round(bm[k]['recall']*14) for k in keys]
false = [round(bm[k]['FPR']*58) for k in keys]
for dy, vals, color, label in [(-.18, detected, teal, '检出负收益（共14项）'), (.18, false, red, '误报正收益（共58项）')]:
    bars = ax.barh(ys+dy, vals, .32, color=color, label=label, zorder=3)
    for b, val in zip(bars, vals):
        ax.text(val+.4, b.get_y()+b.get_height()/2, str(val), va='center', fontsize=15)
ax.set_yticks(ys, labels)
ax.invert_yaxis()
ax.set_xlim(0, 24)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_xlabel('结果数量')
ax.set_title('SC-INR：预测能否超过简单基线？', fontsize=15, pad=17)
ax.grid(axis='x', alpha=.15, zorder=0)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=11, loc='upper right', bbox_to_anchor=(.98, .835), ncol=2)
harm = [r for r in a['negative']['rows'] if r['model']=='sc-inr' and r['shave']==4 and r['bands']=='fullRGB' and r['gain'] < -1e-8]
drops = [10*np.log10(r['native_mse']/r['point_mse']) for r in harm]
fig.text(.07, .17, f'SC的5项整体负收益：PSNR下降{min(drops):.4f}–{max(drops):.4f} dB；裁边0/4/8像素时负例为6/5/5项。', fontsize=12, color=grey)
fig.text(.07, .105, '未解决：加入方向后仍只检出6/14，误报由15增至17；代理策略MSE仅降0.32%，95%区间包含零。', fontsize=12, color=red)
fig.text(.07, .06, '不同倍率、频带共享12张图像，并非独立样本；LTE只有1个频带负例，尚不足以验证稳定的负收益预测。', fontsize=11, color=grey)
fig.savefig(p / 'negative-readout.png', dpi=160)
plt.close(fig)
print(json.dumps({'outputs': ['frequency-recovery.png', 'negative-readout.png'], 'new_experiments': 0}, ensure_ascii=False))
