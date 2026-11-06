import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C1, C2, INK, INK2, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9.5, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                     'lines.linewidth': 2, 'savefig.dpi': 300})
OUT = str(ROOT/'outputs'/'model')+'/'
def L(g, a=.3, mu=.5, se=.4, su=.4, c=0.):
    k = a * (1 - g)
    return (g * mu / (1 - g))**2 + a**2 * ((1 - g)**2 * se**2 + g**2 * su**2 + 2 * g * (1 - g) * c) / (k * (2 - k))

# Figure 1: (a) long-run loss and optimal coupling; (b) admissible region
g = np.linspace(0, .9, 900)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
for mu, col, lab in [(.5, C1, 'μ = 0.5'), (.25, C2, 'μ = 0.25')]:
    y = L(g, mu=mu); i = y[:400].argmin()
    ax[0].plot(g, y, color=col, label=lab); ax[0].plot(g[i], y[i], 'o', color=col, ms=5)
ax[0].set_ylim(0.02, 0.05); ax[0].set_xlim(0, .4)
ax[0].set_xlabel('Self-referential coupling γ'); ax[0].set_ylabel('Long-run loss L(γ)')
ax[0].set_title('(a) Optimal coupling γ° > 0', fontsize=9.5, color=INK)
ax[0].legend(frameon=False, loc='upper left')
for mu, col, lab in [(.5, C1, 'μ = 0.5'), (.25, C2, 'μ = 0.25')]:
    ax[1].plot(g, np.sqrt(L(g, mu=mu)), color=col, label=lab)
ax[1].fill_between(g, np.sqrt(L(g)), 2.0, color=C1, alpha=.08, lw=0)
ax[1].set_xlim(0, .9); ax[1].set_ylim(0, 2)
ax[1].set_xlabel('Self-referential coupling γ'); ax[1].set_ylabel('Size of regime shift ‖Δ‖')
ax[1].set_title('(b) Admissible region (shaded)', fontsize=9.5, color=INK)
ax[1].legend(frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(OUT + 'fig1_v4.png'); plt.close(fig)

# Figure 3: oversight under two attention laws
e = np.linspace(.01, 1, 400); N, c, q0 = 100, 20, .8
hu = q0 * np.minimum(e, c / N)
ho = np.where(e * N <= c, q0 * e, q0 * (c / N) * (c / (e * N))**0.5)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
ax[0].plot(e, hu, color=C1, label='uniform attention'); ax[0].plot(e, ho, color=C2, label='overload decline')
ax[0].set_title(r'Effective control $h_H(e)$', fontsize=9.5, color=INK); ax[0].legend(frameon=False, loc='lower right'); ax[0].set_ylim(0, .185)
ax[1].plot(e, e - hu, color=C1, label='uniform attention'); ax[1].plot(e, e - ho, color=C2, label='overload decline')
ax[1].set_title(r'Decision-level gap $\Psi_D(e)$, with $g(e)=e$', fontsize=9.5, color=INK); ax[1].legend(frameon=False, loc='upper left')
for a in ax:
    a.axvline(c / N, color=INK2, ls=':', lw=1); a.set_xlabel('Escalation share e')
ax[0].text(c / N + .02, .168, 'capacity c/N', fontsize=8.5, color=INK2)
fig.tight_layout(); fig.savefig(OUT + 'fig3_v4.png'); plt.close(fig)
print('ok')
