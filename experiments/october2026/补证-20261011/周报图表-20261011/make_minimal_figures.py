"""Minimal plots of the same saved weekly-report results; no new experiments."""
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
    'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.unicode_minus': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
a = json.loads((p / 'robustness-input.json').read_text())
n = json.loads((p / 'negative-input.json').read_text())
blue, teal, red = '#496780', '#168579', '#CB635B'
rows = [r for r in a['l1']['rows'] if r['seed'] == 1 and r['pooling'] == 'energy_pool'
        and r['domain'] == 'all' and r['scales'] == [2, 3, 4]]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
fig.subplots_adjust(left=.075, right=.98, bottom=.15, top=.78, wspace=.14)
for ax, ds in zip(axes, ['Set14', 'BSD100', 'Urban100']):
    rr = [next(r for r in rows if r['dataset'] == ds and r['model'] == m) for m in ['liif', 'lte', 'sc-inr']]
    x = np.arange(3)
    for dx, key, color, label in [(-.18, 'D_nu', blue, '原始频率'), (.18, 'D_u', teal, '倍率归一化')]:
        vals = [r[key] for r in rr]
        bars = ax.bar(x + dx, vals, .34, color=color, label=label, zorder=3)
        for b, val in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, val+.009, f'{val:.3f}', ha='center', fontsize=11)
    ax.axhline(.05, color=red, ls='--', lw=1.2, label='D = 0.05')
    ax.set_title(ds, fontsize=17, pad=13)
    ax.set_xticks(x, ['LIIF', 'LTE', 'SC-INR'])
    ax.set_ylim(0, .52)
    ax.set_yticks([0, .1, .2, .3, .4, .5])
    ax.grid(axis='y', alpha=.15, zorder=0)
axes[0].set_ylabel('跨倍率差异 D')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.53, 1.0))
fig.savefig(p / 'frequency-recovery-minimal.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={'width_ratios': [1, 1.2]})
fig.subplots_adjust(left=.075, right=.98, bottom=.18, top=.77, wspace=.40)
ax = axes[0]
x = np.arange(2)
for dx, key, denom, color, label in [(-.18, 'band_metrics', 72, blue, '频带 (n=72)'), (.18, 'RGB_policies', 24, red, '整体 (n=24)')]:
    counts = [n['models'][m][key]['always_enable']['negative'] for m in ['sc-inr', 'lte']]
    vals = [100*c/denom for c in counts]
    bars = ax.bar(x+dx, vals, .34, color=color, label=label, zorder=3)
    for b, c, val in zip(bars, counts, vals):
        ax.text(b.get_x()+b.get_width()/2, val+1, f'{c}/{denom}', ha='center', fontsize=14)
ax.set_xticks(x, ['SC-INR', 'LTE'])
ax.set_ylim(0, 30)
ax.set_ylabel('负收益比例 (%)')
ax.set_title('负收益', fontsize=17, pad=15)
ax.grid(axis='y', alpha=.15, zorder=0)
ax.legend(frameon=False, fontsize=12, loc='upper right')
ax = axes[1]
keys = ['norm_ridge', 'directional_ridge', 'proxy']
ys = np.arange(3)
bm = n['models']['sc-inr']['band_metrics']
detected = [round(bm[k]['recall']*14) for k in keys]
false = [round(bm[k]['FPR']*58) for k in keys]
for dy, vals, color, label in [(-.18, detected, teal, '检出 (n=14)'), (.18, false, red, '误报 (n=58)')]:
    bars = ax.barh(ys+dy, vals, .32, color=color, label=label, zorder=3)
    for b, val in zip(bars, vals):
        ax.text(val+.4, b.get_y()+b.get_height()/2, str(val), va='center', fontsize=14)
ax.set_yticks(ys, ['幅度/频谱', '＋方向', '双三次代理'])
ax.invert_yaxis()
ax.set_xlim(0, 21)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_xlabel('数量')
ax.set_title('预测对比：SC-INR', fontsize=17, pad=15)
ax.grid(axis='x', alpha=.15, zorder=0)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=12, loc='upper right', bbox_to_anchor=(.98, .99), ncol=2)
fig.savefig(p / 'negative-readout-minimal.png', dpi=180)
plt.close(fig)
manifest = {'scope': 'Presentation revision only; same saved inputs and primary scoring.',
    'inputs': [{'file': f, 'sha256': hashlib.sha256((p/f).read_bytes()).hexdigest()} for f in ['robustness-input.json', 'negative-input.json']],
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'outputs': [{'file': f, 'sha256': hashlib.sha256((p/f).read_bytes()).hexdigest()} for f in ['frequency-recovery-minimal.png', 'negative-readout-minimal.png']]}
(p/'minimal-sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print('Rendered two minimal plots; no training, inference or fitting.')
