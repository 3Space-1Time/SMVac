"""
replot_scatter.py — Redesign Figure 4 (S_exact vs S_approx) as a clean density+scatter plot.
Uses hexbin density for the full 252k dataset, then overlays reference lines and marginal info.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

DARK   = '#1a1a2e'
BLUE   = '#4FC3F7'
GREEN  = '#81C784'
RED    = '#EF5350'
GOLD   = '#FFD54F'
WHITE  = '#E0E0E0'
PURPLE = '#CE93D8'

# ── Load ALL valid data ───────────────────────────────────────────────────────
print("Loading data...")
num = pd.read_csv('results/numerical_data.csv')
valid = num[
    (num['Stability'].isin([2, 3])) &
    (num['S_exact']  > 0) & (num['S_exact']  < 1e50) &
    (num['S_approx'] > 0) & (num['S_approx'] < 1e50)
].copy()
print(f"  Valid points: {len(valid):,}")

# Work in log10 space — much cleaner for scatter
valid['log_ap'] = np.log10(valid['S_approx'])
valid['log_ex'] = np.log10(valid['S_exact'])
valid['rel_err'] = (valid['S_exact'] - valid['S_approx']) / valid['S_exact'] * 100.0

# Clip dynamic range so the plot is meaningful
# S > 1e5 is deep in metastable territory and contains ceiling artifacts
clip_lo, clip_hi = 0.5, 5.5   # log10 units
mask = (valid['log_ap'].between(clip_lo, clip_hi) &
        valid['log_ex'].between(clip_lo, clip_hi) &
        (np.abs(valid['log_ex'] - valid['log_ap']) < 1.5))   # exclude |delta_log| > 1.5 artifacts
df = valid[mask].copy()
meta = df[df['Stability'] == 2]
unst = df[df['Stability'] == 3]
print(f"  After clip: {len(df):,}  (meta={len(meta):,}, unstable={len(unst):,})")

# ── Figure layout: main scatter + right residual strip ────────────────────────
fig = plt.figure(figsize=(11, 7), facecolor=DARK)
gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.04, figure=fig)
ax  = fig.add_subplot(gs[0])   # main panel
axr = fig.add_subplot(gs[1])   # residual histogram on the right
for a in [ax, axr]:
    a.set_facecolor(DARK)

# ── MAIN PANEL ─────────────────────────────────────────────────────────────
# 1. Hexbin density background (all points together)
hb = ax.hexbin(df['log_ap'], df['log_ex'],
               gridsize=120,
               cmap='plasma',
               mincnt=1,
               linewidths=0.0,
               bins='log')   # log-scale counts → colour by log10(N)

cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.03)
cb.set_label('log$_{10}$(point count per bin)', color=WHITE, fontsize=10)
cb.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)

# 2. Perfect-agreement diagonal
lim_lo, lim_hi = clip_lo, clip_hi
diag = np.linspace(lim_lo, lim_hi, 300)
ax.plot(diag, diag, '--', color=WHITE, lw=2.0, zorder=6, label='$S_{exact} = S_{approx}$')

# 3. ±5% and ±15% envelopes (in log space: log10(1±x) shifts)
for pct, ls, label in [(0.05, ':', '±5%'), (0.15, '--', '±15%')]:
    ax.plot(diag, diag + np.log10(1 + pct), ls, color=GOLD, lw=1.3, zorder=5, alpha=0.8)
    ax.plot(diag, diag + np.log10(1 - pct), ls, color=GOLD, lw=1.3, zorder=5, alpha=0.8,
            label=label if pct == 0.05 else '_nolegend_')
for pct, ls in [(0.15, '--')]:
    ax.plot(diag, diag + np.log10(1 + pct), ls, color=PURPLE, lw=1.1, zorder=5, alpha=0.7)
    ax.plot(diag, diag + np.log10(max(0.01, 1 - pct)), ls, color=PURPLE, lw=1.1, zorder=5, alpha=0.7,
            label='±15%')

# 4. Thin boundary-scatter: only points within the ±5% envelope strip
# (outer layer — shows where disagreement is largest)
outliers = df[np.abs(df['log_ex'] - df['log_ap']) > np.log10(1.05)]
ax.scatter(outliers['log_ap'], outliers['log_ex'],
           c=np.where(outliers['Stability']==2, GREEN, RED),
           s=2.5, alpha=0.6, zorder=7, rasterized=True,
           label=f'Outliers >5% ({len(outliers):,})')

# 5. Axes labels using plain tick values but show GeV-free numbers
ax.set_xlim(clip_lo, clip_hi)
ax.set_ylim(clip_lo, clip_hi)

xticks = [1, 2, 3, 4, 5]
ax.set_xticks(xticks)
ax.set_xticklabels([f'$10^{{{int(x)}}}$' for x in xticks], color=WHITE)
ax.set_yticks(xticks)
ax.set_yticklabels([f'$10^{{{int(x)}}}$' for x in xticks], color=WHITE)

ax.set_xlabel(r'$S_{approx}$ — Conformal analytical approximation  $\frac{16\pi^2}{3|\lambda_{min}|}$',
              color=WHITE)
ax.set_ylabel(r'$S_{exact}$ — Full numerical $\phi^6$ bounce integration', color=WHITE)
ax.set_title(r'Bounce Action: Exact Numerical vs. Conformal Approximation'
             '\n' + rf'({len(df):,} points; coloured by local density)',
             color=WHITE, fontweight='bold')

ax.tick_params(colors=WHITE, which='both')
for sp in ax.spines.values():
    sp.set_color('#444')
ax.grid(True, color='#333', ls='--', lw=0.5, alpha=0.5)

# Custom legend
legend_elements = [
    Line2D([0],[0], color=WHITE, ls='--', lw=2, label='$S_{exact}=S_{approx}$'),
    Line2D([0],[0], color=GOLD,  ls=':',  lw=1.5, label='±5% envelope'),
    Line2D([0],[0], color=PURPLE,ls='--', lw=1.2, label='±15% envelope'),
    Line2D([0],[0], color=GREEN, marker='o', ms=5, ls='none', label='Metastable outliers'),
    Line2D([0],[0], color=RED,   marker='o', ms=5, ls='none', label='Unstable outliers'),
]
ax.legend(handles=legend_elements, facecolor='#1e1e38', edgecolor='#555',
          labelcolor=WHITE, loc='upper left', framealpha=0.85, fontsize=9.5)

# ── RIGHT PANEL: Relative error distribution ──────────────────────────────────
rel = df['rel_err'].clip(-20, 20)   # clip extreme tails
bins = np.linspace(-20, 20, 100)

# Split by stability
axr.hist(meta['rel_err'].clip(-30,30), bins=bins, orientation='horizontal',
         color=GREEN, alpha=0.65, density=True, label='Metastable')
axr.hist(unst['rel_err'].clip(-30,30), bins=bins, orientation='horizontal',
         color=RED,   alpha=0.65, density=True, label='Unstable')

axr.axhline(0, color=WHITE, ls='--', lw=1.5)
axr.axhline(+5, color=GOLD, ls=':', lw=1.2)
axr.axhline(-5, color=GOLD, ls=':', lw=1.2)

med_meta = meta['rel_err'].median()
med_unst = unst['rel_err'].median()
axr.axhline(med_meta, color=GREEN, ls='-',  lw=1.5, alpha=0.8)
axr.axhline(med_unst, color=RED,   ls='-',  lw=1.5, alpha=0.8)

axr.set_ylim(-20, 20)
axr.set_xlabel('Density', color=WHITE)
axr.set_title('Residual\ndistribution\n[%]', color=WHITE, fontsize=10)
axr.tick_params(colors=WHITE, labelleft=False, labelright=True)
axr.yaxis.set_label_position('right')
axr.yaxis.tick_right()
axr.set_ylabel(r'$(S_{exact}-S_{approx})/S_{exact}$ [%]', color=WHITE, fontsize=9)
for sp in axr.spines.values():
    sp.set_color('#444')
axr.set_facecolor(DARK)
axr.grid(True, color='#333', ls='--', lw=0.5)

axr.legend(facecolor='#1e1e38', edgecolor='#555', labelcolor=WHITE,
           fontsize=8, loc='upper right')

# Annotate medians
axr.annotate(f'Median\n{med_meta:+.2f}%', xy=(axr.get_xlim()[1]*0.6, med_meta+1.5),
             color=GREEN, fontsize=8)
axr.annotate(f'Median\n{med_unst:+.2f}%', xy=(axr.get_xlim()[1]*0.6, med_unst-3.5),
             color=RED, fontsize=8)

plt.savefig('results/error_S_scatter.png', dpi=200,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("Saved results/error_S_scatter.png")
