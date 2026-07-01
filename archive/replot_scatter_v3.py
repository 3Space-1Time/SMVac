"""
replot_scatter_v3.py — Final S_exact vs S_approx scatter with correct range, annotations,
and clear explanation of the two physical populations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    'legend.fontsize': 9.5,
})

DARK   = '#1a1a2e'
BLUE   = '#4FC3F7'
GREEN  = '#81C784'
RED    = '#EF5350'
GOLD   = '#FFD54F'
WHITE  = '#E0E0E0'
PURPLE = '#CE93D8'
ORANGE = '#FFA726'

# ── Load ALL valid data ───────────────────────────────────────────────────────
print("Loading data...")
num = pd.read_csv('results/numerical_data.csv')
an  = pd.read_csv('results/analytical_data.csv')

valid = num[
    (num['Stability'].isin([2, 3])) &
    (num['S_exact']  > 0) & (num['S_exact']  < 1e50) &
    (num['S_approx'] > 0) & (num['S_approx'] < 1e50)
].copy()
valid['log_ap'] = np.log10(valid['S_approx'])
valid['log_ex'] = np.log10(valid['S_exact'])
valid['delta']  = valid['log_ex'] - valid['log_ap']
print(f"  Valid points: {len(valid):,}")

# Full range — do NOT clip artificially
clip_lo, clip_hi = 0.3, 8.5
df = valid[valid['log_ap'].between(clip_lo, clip_hi) & valid['log_ex'].between(clip_lo, clip_hi)].copy()

# Classify populations
main_diag = df[np.abs(df['delta']) <= 0.15]           # tight diagonal (±5% in linear space ≈ ±0.02 log)
phi6_cluster = df[(df['delta'] > 0.15) & (df['Stability']==2)]  # phi6-boosted metastable
phi6_unstable= df[(df['delta'] > 0.15) & (df['Stability']==3)]
below_diag   = df[df['delta'] < -0.15]

print(f"  Main diagonal (|delta|<=0.15 log): {len(main_diag):,}")
print(f"  phi6 cluster above diagonal (metastable): {len(phi6_cluster):,}")
print(f"  phi6 cluster above diagonal (unstable):   {len(phi6_unstable):,}")
print(f"  Below diagonal:  {len(below_diag):,}")

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 8), facecolor=DARK)
gs  = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05, figure=fig)
ax  = fig.add_subplot(gs[0])
axr = fig.add_subplot(gs[1])
for a in [ax, axr]:
    a.set_facecolor(DARK)

# ── MAIN DIAGONAL population: hexbin density ─────────────────────────────────
hb = ax.hexbin(main_diag['log_ap'], main_diag['log_ex'],
               gridsize=130, cmap='plasma', mincnt=1,
               linewidths=0.0, bins='log', zorder=2)
cb = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.03)
cb.set_label('Density  log$_{10}$(count/bin)', color=WHITE, fontsize=9.5)
cb.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)

# ── PHI6 CLUSTER — metastable points high above diagonal ────────────────────
ax.scatter(phi6_cluster['log_ap'], phi6_cluster['log_ex'],
           c=GREEN, s=3.5, alpha=0.55, zorder=4, rasterized=True,
           label=f'$\\phi^6$-boosted metastable  ({len(phi6_cluster):,} pts)')
ax.scatter(phi6_unstable['log_ap'], phi6_unstable['log_ex'],
           c=ORANGE, s=2.5, alpha=0.45, zorder=3, rasterized=True,
           label=f'$\\phi^6$-boosted unstable  ({len(phi6_unstable):,} pts)')
ax.scatter(below_diag['log_ap'], below_diag['log_ex'],
           c=RED, s=2.0, alpha=0.35, zorder=3, rasterized=True,
           label=f'Below diagonal  ({len(below_diag):,} pts)')

# ── Reference lines ───────────────────────────────────────────────────────────
diag = np.linspace(clip_lo, clip_hi, 400)
ax.plot(diag, diag,             '--', color=WHITE,  lw=2.0, zorder=6,
        label='$S_{exact} = S_{approx}$')
ax.plot(diag, diag+np.log10(1.05), ':', color=GOLD, lw=1.3, zorder=5, alpha=0.85,
        label='±5%')
ax.plot(diag, diag-np.log10(1.05), ':', color=GOLD, lw=1.3, zorder=5, alpha=0.85)
ax.plot(diag, diag+np.log10(2.0), ':', color=PURPLE, lw=1.2, zorder=5, alpha=0.75,
        label='×2 / ÷2')
ax.plot(diag, diag-np.log10(2.0), ':', color=PURPLE, lw=1.2, zorder=5, alpha=0.75)

# ── Annotation arrows for the two key features ───────────────────────────────
# 1. Cluster of phi6-boosted metastable points
ax.annotate(
    'Dense $\\phi^6$-boosted\nmetastable cluster\n'
    r'$S_{exact} \gg S_{approx}$' + '\n' +
    r'$S_{exact}/S_{approx}$: median ~113×' + '\n'
    '(shallow instability regime)',
    xy=(3.7, 6.0), xytext=(6.2, 5.5),
    color=GREEN, fontsize=9, ha='left',
    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.4, connectionstyle='arc3,rad=-0.2'),
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1127', edgecolor=GREEN, alpha=0.85)
)

# 2. Why no points on diagonal above 10^4 → because those 2154 points
#    all have delta > 0 and end up in the green cluster
ax.annotate(
    'Points with\n$S_{approx}>10^4$\n'
    'all jump to\nthis band\n(2,154 pts)',
    xy=(4.5, 7.2), xytext=(2.5, 7.8),
    color=GOLD, fontsize=8.5, ha='center',
    arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1127', edgecolor=GOLD, alpha=0.85)
)

# 3. Main diagonal label
ax.annotate(
    '~242k points\ntight diagonal\n(|error|<5%)',
    xy=(2.0, 2.1), xytext=(0.6, 4.5),
    color=WHITE, fontsize=8.5, ha='center',
    arrowprops=dict(arrowstyle='->', color=WHITE, lw=1.2, connectionstyle='arc3,rad=0.3'),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1127', edgecolor='#555', alpha=0.85)
)

# ── Axis formatting ───────────────────────────────────────────────────────────
ax.set_xlim(clip_lo, clip_hi)
ax.set_ylim(clip_lo, clip_hi)

xticks = [1, 2, 3, 4, 5, 6, 7, 8]
ax.set_xticks(xticks)
ax.set_xticklabels([f'$10^{{{x}}}$' for x in xticks], color=WHITE)
ax.set_yticks(xticks)
ax.set_yticklabels([f'$10^{{{x}}}$' for x in xticks], color=WHITE)

ax.set_xlabel(r'$S_{approx}$ — Conformal  $\frac{16\pi^2}{3|\lambda_{min}|}$', color=WHITE)
ax.set_ylabel(r'$S_{exact}$ — Full numerical $\phi^6$ bounce integration', color=WHITE)
ax.set_title(
    r'Bounce Action: $S_{exact}$ vs $S_{approx}$  —  Two Distinct Physical Populations'
    '\n' + rf'Main diagonal: {len(main_diag):,} pts   |   '
    rf'$\phi^6$-cluster: {len(phi6_cluster):,} pts   |   total: {len(df):,} pts',
    color=WHITE, fontweight='bold')

ax.tick_params(colors=WHITE, which='both')
for sp in ax.spines.values():
    sp.set_color('#444')
ax.grid(True, color='#2a2a4a', ls='--', lw=0.5)

legend_elements = [
    Line2D([0],[0], color=WHITE,  ls='--', lw=2,   label='$S_{exact}=S_{approx}$ (exact agreement)'),
    Line2D([0],[0], color=GOLD,   ls=':',  lw=1.4, label='±5% envelope'),
    Line2D([0],[0], color=PURPLE, ls=':',  lw=1.2, label='×2 / ÷2 envelope'),
    Line2D([0],[0], color=GREEN,  marker='o', ms=5, ls='none',
           label=rf'$\phi^6$-boosted metastable ({len(phi6_cluster):,})'),
    Line2D([0],[0], color=ORANGE, marker='o', ms=5, ls='none',
           label=rf'$\phi^6$-boosted unstable ({len(phi6_unstable):,})'),
    Line2D([0],[0], color=RED,    marker='o', ms=5, ls='none',
           label=rf'Below-diagonal ({len(below_diag):,})'),
]
ax.legend(handles=legend_elements, facecolor='#1e1e38', edgecolor='#555',
          labelcolor=WHITE, loc='upper left', framealpha=0.88, fontsize=8.5)

# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
bins = np.linspace(-25, 25, 100)
axr.hist(df[df['Stability']==2]['delta'].clip(-25,25)*10, bins=bins,   # *10 to convert log10 to ~%, approx
         orientation='horizontal', color=GREEN,  alpha=0.65, density=True, label='Metastable')
axr.hist(df[df['Stability']==3]['delta'].clip(-25,25)*10, bins=bins,
         orientation='horizontal', color=RED,    alpha=0.55, density=True, label='Unstable')

axr.axhline(0,  color=WHITE, ls='--', lw=1.5)
axr.axhline( 1, color=GOLD,  ls=':',  lw=1.0, alpha=0.8, label='log ratio = ±0.1')
axr.axhline(-1, color=GOLD,  ls=':',  lw=1.0, alpha=0.8)
axr.axhline(3,  color=PURPLE,ls=':',  lw=1.0, alpha=0.8, label='log ratio = ±0.3 (×2)')
axr.axhline(-3, color=PURPLE,ls=':',  lw=1.0, alpha=0.8)

axr.set_ylim(-8, 25)
axr.set_xlabel('Density', color=WHITE)
axr.set_title('$\\log_{10}(S_{ex}/S_{ap})$\ndistribution\n(×10)', color=WHITE, fontsize=9.5)
axr.tick_params(colors=WHITE, labelleft=False, labelright=True)
axr.yaxis.set_label_position('right')
axr.yaxis.tick_right()
axr.set_ylabel(r'$10\cdot\log_{10}(S_{exact}/S_{approx})$', color=WHITE, fontsize=8.5)
for sp in axr.spines.values():
    sp.set_color('#444')
axr.set_facecolor(DARK)
axr.grid(True, color='#2a2a4a', ls='--', lw=0.5)
axr.legend(facecolor='#1e1e38', edgecolor='#555', labelcolor=WHITE, fontsize=8, loc='upper right')

# Annotate the two populations
axr.annotate('Main\ndiagonal\npopulation', xy=(0.6, 0.2), color=WHITE, fontsize=7.5)
axr.annotate('$\\phi^6$\ncluster', xy=(0.6, 12), color=GREEN, fontsize=7.5)

plt.savefig('results/error_S_scatter.png', dpi=200,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("Saved results/error_S_scatter.png")
print(f"\nPhysical interpretation:")
print(f"  phi6 cluster: median S_exact/S_approx = {(10**df[df['delta']>0.15]['delta']).median():.1f}x")
print(f"  Largest ratio: {(10**df['delta'].max()):.2e}x")
