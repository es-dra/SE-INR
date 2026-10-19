"""Analytic figure only; no fitted experimental results or target data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
def delta(r, k):
    return r * (1-r)**2 * (k-1)**2 / (1-r+k*r)**2

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)
r = np.linspace(.001, .999, 1000)
axes[0].plot(r, delta(r, 4), color='#176B87', label=r'Overestimate: $\kappa=4$')
axes[0].plot(r, delta(r, .25), color='#B65C45', ls='--', label=r'Underestimate: $\kappa=1/4$')
axes[0].axvline(.5, color='0.6', ls=':', lw=1)
axes[0].set(xlabel=r'Scalar matched recovery $R$', ylabel=r'Normalized excess MSE $\Delta$',
            title='(a) Exact scalar asymmetry', xlim=(0,1), ylim=(0,.9))
axes[0].legend(frameon=False, fontsize=8)
exact = [delta(np.array([.1,.9]), 4).mean(), delta(np.array([.01,.99]), 4).mean()]
x = np.arange(2)
axes[1].bar(x-.17, exact, .32, color='#176B87', label='Average exact costs')
axes[1].bar(x+.17, [.18,.18], .32, color='#B65C45', label='Cost of averaged R')
axes[1].set_xticks(x, [r'$R=(.1,.9)$', r'$R=(.01,.99)$'])
axes[1].set(ylabel='Equal-energy average excess MSE', title='(b) Aggregation error changes sign', ylim=(0,.28))
axes[1].legend(frameon=False, fontsize=8)
for i,v in enumerate(exact):
    axes[1].text(i-.17,v+.008,f'{v:.3f}',ha='center',fontsize=8)
for ax in axes:
    ax.grid(axis='y',color='0.92',lw=.6)
    ax.set_axisbelow(True)
fig.savefig(OUT/'scalar-cost.pdf')
fig.savefig(OUT/'scalar-cost.png', dpi=200)
