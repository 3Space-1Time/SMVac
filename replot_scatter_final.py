"""
replot_scatter_final.py — Clean 3-panel Figure 4.
Panel A: S_exact vs S_approx scatter (physically relevant range, S = 3-10^4)
Panel B: Relative error (%) vs S_approx — shows trend and breakdown
Panel C: Zoomed scatter near the metastability threshold
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 10.5,
    'axes.titlesize': 11.5,
    'axes.labelsize': 10.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
})

DARK   = '#12122a'
BLUE   = '#4FC3F7'
GREEN  = '#66BB6A'
RED    = '#EF5350'
GOLD   = '#FFD54F'
WHITE  = '#DEDEDE'
PURPLE = '#CE93D8'
GREY   = '#555577'

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading...")
num = pd.read_csv('results/numerical_data.csv')
valid = num[
    (num['Stability'].isin([2, 3])) &
    (num['S_exact']  > 1) & (num['S_exact']  < 1e50) &
    (num['S_approx'] > 1) & (num['S_approx'] < 1e8)
].copy()
valid['rel_pct'] = (valid['S_exact'] - valid['S_approx']) / valid['S_exact'] * 100.0
meta = valid[valid['Stability'] == 2]
unst = valid[valid['Stability'] == 3]
print(f"  Meta: {len(meta):,}  Unstable: {len(unst):,}")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5.5), facecolor=DARK)
fig.suptitle(
    r'Bounce Action Comparison: Exact Numerical ($\phi^6$) vs. Conformal Analytical Approximation',
    color=WHITE, fontsize=13, fontweight='bold', y=1.01
)
gs = gridspec.GridSpec(1, 3, wspace=0.38, figure=fig)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])

for ax in [axA, axB, axC]:
    ax.set_facecolor(DARK)
    ax.tick_params(colors=WHITE, which='both')
    for sp in ax.spines.values():
        sp.set_color('#444')
    ax.grid(True, color='#2a2a42', ls='--', lw=0.6)

# ════════════════════════════════════════════════════════════════════════════
# PANEL A: Full log-log scatter, S = 3 to 10^4
# ════════════════════════════════════════════════════════════════════════════
# Sample for clarity — 20k each is plenty, main story is the reference lines
rng = np.random.default_rng(42)
m_samp = meta[meta['S_approx'] < 1e4].sample(min(25000, len(meta[meta['S_approx']<1e4])), random_state=42)
u_samp = unst[unst['S_approx'] < 1e4].sample(min(25000, len(unst[unst['S_approx']<1e4])), random_state=42)

axA.scatter(m_samp['S_approx'], m_samp['S_exact'],
            c=GREEN, s=1.2, alpha=0.18, rasterized=True, zorder=2)
axA.scatter(u_samp['S_approx'], u_samp['S_exact'],
            c=RED,   s=1.2, alpha=0.18, rasterized=True, zorder=2)

# Reference lines
S_ref = np.logspace(0.5, 4.0, 500)
axA.plot(S_ref, S_ref, '--', color=WHITE, lw=1.8, zorder=5,
         label=r'$S_{exact}=S_{approx}$')
axA.fill_between(S_ref, S_ref * 0.95, S_ref * 1.05,
                 color=GOLD, alpha=0.12, zorder=3, label='±5% band')
axA.plot(S_ref, S_ref * 1.05, '-', color=GOLD,  lw=0.8, alpha=0.6, zorder=4)
axA.plot(S_ref, S_ref * 0.95, '-', color=GOLD,  lw=0.8, alpha=0.6, zorder=4)
axA.fill_between(S_ref, S_ref * 0.85, S_ref * 1.15,
                 color=PURPLE, alpha=0.08, zorder=3, label='±15% band')
axA.plot(S_ref, S_ref * 1.15, '-', color=PURPLE, lw=0.8, alpha=0.5, zorder=4)
axA.plot(S_ref, S_ref * 0.85, '-', color=PURPLE, lw=0.8, alpha=0.5, zorder=4)

axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xlim(3, 1e4); axA.set_ylim(3, 1e4)
axA.set_xlabel(r'$S_{approx} = \frac{16\pi^2}{3|\lambda_{min}|}$', color=WHITE)
axA.set_ylabel(r'$S_{exact}$ (numerical $\phi^6$ integration)', color=WHITE)
axA.set_title('(a)  Full Scatter — Physically Relevant Range', color=WHITE, fontweight='bold')

legend_elems = [
    Line2D([0],[0], color=WHITE, ls='--', lw=1.8, label=r'$S_{exact}=S_{approx}$'),
    Patch(facecolor=GOLD,   alpha=0.4,  label='±5% band'),
    Patch(facecolor=PURPLE, alpha=0.35, label='±15% band'),
    Line2D([0],[0], color=GREEN, marker='o', ms=5, ls='none', label='Metastable', alpha=0.8),
    Line2D([0],[0], color=RED,   marker='o', ms=5, ls='none', label='Unstable',   alpha=0.8),
]
axA.legend(handles=legend_elems, facecolor='#1a1a35', edgecolor='#444',
           labelcolor=WHITE, loc='upper left', framealpha=0.9)

# Count points within bands
in5pct  = ((valid['S_exact'] < valid['S_approx']*1.05) &
            (valid['S_exact'] > valid['S_approx']*0.95) &
            (valid['S_approx'] < 1e4))
in15pct = ((valid['S_exact'] < valid['S_approx']*1.15) &
            (valid['S_exact'] > valid['S_approx']*0.85) &
            (valid['S_approx'] < 1e4))
total_range = valid[valid['S_approx'] < 1e4]
axA.text(0.97, 0.07,
         f"{in5pct.sum()/len(total_range)*100:.1f}% within ±5%\n"
         f"{in15pct.sum()/len(total_range)*100:.1f}% within ±15%",
         transform=axA.transAxes, color=GOLD, fontsize=9,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='#1a1a35', edgecolor='#666', alpha=0.85))

# ════════════════════════════════════════════════════════════════════════════
# PANEL B: Relative error (%) vs S_approx — binned mean + scatter
# ════════════════════════════════════════════════════════════════════════════
# Only use S < 5000 range where the error structure is physically meaningful
work = valid[(valid['S_approx'].between(3, 5000)) &
             (valid['rel_pct'].between(-50, 50))].copy()

# Scatter (thin, background)
axB.scatter(work[work['Stability']==3]['S_approx'],
            work[work['Stability']==3]['rel_pct'],
            c=RED,   s=0.8, alpha=0.08, rasterized=True, zorder=2)
axB.scatter(work[work['Stability']==2]['S_approx'],
            work[work['Stability']==2]['rel_pct'],
            c=GREEN, s=0.8, alpha=0.10, rasterized=True, zorder=2)

# Binned mean ± std — the real story
log_bins = np.logspace(np.log10(3), np.log10(5000), 50)
bin_centers, bin_mean, bin_std = [], [], []
for i in range(len(log_bins)-1):
    mask = work['S_approx'].between(log_bins[i], log_bins[i+1])
    sub  = work[mask]['rel_pct']
    if len(sub) > 20:
        bin_centers.append(np.sqrt(log_bins[i]*log_bins[i+1]))
        bin_mean.append(sub.mean())
        bin_std.append(sub.std())

bin_centers = np.array(bin_centers)
bin_mean    = np.array(bin_mean)
bin_std     = np.array(bin_std)

axB.plot(bin_centers, bin_mean, '-', color=BLUE, lw=2.2, zorder=6,
         label='Binned mean')
axB.fill_between(bin_centers,
                 bin_mean - bin_std, bin_mean + bin_std,
                 color=BLUE, alpha=0.2, zorder=5, label=r'$\pm 1\sigma$')

axB.axhline(0,   color=WHITE, ls='--', lw=1.4, zorder=7)
axB.axhline(+5,  color=GOLD,  ls=':',  lw=1.0, alpha=0.8)
axB.axhline(-5,  color=GOLD,  ls=':',  lw=1.0, alpha=0.8, label='±5%')
axB.axhline(+15, color=PURPLE,ls=':',  lw=1.0, alpha=0.6)
axB.axhline(-15, color=PURPLE,ls=':',  lw=1.0, alpha=0.6, label='±15%')

axB.set_xscale('log')
axB.set_xlim(3, 5000)
axB.set_ylim(-35, 35)
axB.set_xlabel(r'$S_{approx}$', color=WHITE)
axB.set_ylabel(r'$(S_{exact} - S_{approx})\,/\,S_{exact}$ [%]', color=WHITE)
axB.set_title('(b)  Relative Discrepancy vs. Action Value', color=WHITE, fontweight='bold')
axB.legend(facecolor='#1a1a35', edgecolor='#444', labelcolor=WHITE,
           loc='upper right', framealpha=0.9)

# Annotate regime shift
axB.annotate('Conformal approx.\nworks well here\n($S$ large → $|\\lambda_{min}|$ small)',
             xy=(2000, 2), xytext=(600, 22),
             color=GREEN, fontsize=8.5,
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.1),
             bbox=dict(boxstyle='round', facecolor='#1a1a35', edgecolor=GREEN, alpha=0.85))
axB.annotate('Scatter grows\nnear threshold\n($S$ small → large $|\\lambda_{min}|$)',
             xy=(10, 8), xytext=(20, 28),
             color=RED, fontsize=8.5,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.1),
             bbox=dict(boxstyle='round', facecolor='#1a1a35', edgecolor=RED, alpha=0.85))

# ════════════════════════════════════════════════════════════════════════════
# PANEL C: Zoomed — near metastability threshold where classification is made
# ════════════════════════════════════════════════════════════════════════════
# The universe age threshold is S_threshold ~ 4*ln(T_U * mu1)
# Typical S_threshold ~ 400-500 for SM point
# So let's zoom to S = 50–2000

zoom = valid[(valid['S_approx'].between(50, 2000)) &
             (valid['S_exact'].between(50, 2000))].copy()
zm_meta = zoom[zoom['Stability'] == 2].sample(min(15000, len(zoom[zoom['Stability']==2])), random_state=42)
zm_unst = zoom[zoom['Stability'] == 3].sample(min(15000, len(zoom[zoom['Stability']==3])), random_state=42)

axC.scatter(zm_unst['S_approx'], zm_unst['S_exact'],
            c=RED,   s=2.0, alpha=0.3, rasterized=True, zorder=2)
axC.scatter(zm_meta['S_approx'], zm_meta['S_exact'],
            c=GREEN, s=2.0, alpha=0.3, rasterized=True, zorder=2)

# Diagonal and bands
S_z = np.linspace(50, 2000, 300)
axC.plot(S_z, S_z, '--', color=WHITE, lw=1.8, zorder=6)
axC.fill_between(S_z, S_z*0.95, S_z*1.05,
                 color=GOLD, alpha=0.15, zorder=3)
axC.plot(S_z, S_z*1.05, '-', color=GOLD, lw=0.9, alpha=0.7)
axC.plot(S_z, S_z*0.95, '-', color=GOLD, lw=0.9, alpha=0.7)

# Mark a typical S_threshold
S_thr_typical = 450
axC.axvline(S_thr_typical, color=BLUE,  ls='-.', lw=1.4, zorder=7,
            label=f'Typical $S_{{threshold}}\\approx{S_thr_typical}$')
axC.axhline(S_thr_typical, color=BLUE,  ls='-.', lw=1.4, zorder=7)

# Shade the reclassification quadrant
# Bottom-right: S_approx > threshold AND S_exact < threshold → analytical says meta, num says unstable
# Top-left: S_approx < threshold AND S_exact > threshold → analytical says unstable, num says meta
axC.fill_between([50, S_thr_typical], [S_thr_typical, S_thr_typical], [2000, 2000],
                 color=GREEN, alpha=0.07, zorder=1)
axC.fill_between([S_thr_typical, 2000], [50, 50], [S_thr_typical, S_thr_typical],
                 color=RED, alpha=0.07, zorder=1)

axC.text(130, 1700, r'$S_{approx}<S_{thr}$ but $S_{exact}>S_{thr}$'+'\n→ Numerical rescues\n   to Metastable',
         color=GREEN, fontsize=8, va='top',
         bbox=dict(boxstyle='round', facecolor='#1a1a35', edgecolor=GREEN, alpha=0.85))
axC.text(700, 150, r'$S_{approx}>S_{thr}$ but $S_{exact}<S_{thr}$'+'\n→ Numerical downgrades\n   to Unstable',
         color=RED, fontsize=8, va='bottom',
         bbox=dict(boxstyle='round', facecolor='#1a1a35', edgecolor=RED, alpha=0.85))

axC.set_xlim(50, 2000); axC.set_ylim(50, 2000)
axC.set_xlabel(r'$S_{approx}$', color=WHITE)
axC.set_ylabel(r'$S_{exact}$', color=WHITE)
axC.set_title('(c)  Zoomed: Near Metastability Threshold', color=WHITE, fontweight='bold')
axC.legend(facecolor='#1a1a35', edgecolor='#444', labelcolor=WHITE,
           loc='lower right', framealpha=0.9)
axC.set_aspect('equal')

plt.savefig('results/error_S_scatter.png', dpi=200,
            bbox_inches='tight', facecolor=DARK)
plt.close()

# Print stats for Panel C
in5 = ((zoom['S_exact'] < zoom['S_approx']*1.05) & (zoom['S_exact'] > zoom['S_approx']*0.95))
print(f"\nIn zoomed threshold region (S=50-2000): {len(zoom):,} points")
print(f"  Within ±5%: {in5.sum()/len(zoom)*100:.1f}%")
print(f"  Median relative error: {zoom['rel_pct'].median():.3f}%")
print(f"  Binned mean error at S=500: {work[(work['S_approx'].between(450,550))]['rel_pct'].mean():.3f}%")
print("Saved results/error_S_scatter.png")
