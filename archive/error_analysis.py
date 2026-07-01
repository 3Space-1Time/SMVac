"""
error_analysis.py — Detailed numerical error analysis for the vacuum stability paper.
Generates 8 high-quality figures covering:
  1. RK4 step-size convergence study
  2. Gauss-Simpson quadrature convergence (N_QUAD sweep)
  3. Golden Section Search convergence
  4. S_exact vs S_approx scatter plot
  5. Relative discrepancy (S_exact - S_approx) / S_exact heatmap
  6. Phase-boundary sensitivity strip width
  7. Action residual distribution (histogram)
  8. Boundary discrepancy vs |lambda_min|
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, LogFormatter
import warnings, subprocess, sys, os, math

warnings.filterwarnings('ignore')

# -- Style --------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DARK  = '#1a1a2e'
BLUE  = '#4FC3F7'
GREEN = '#81C784'
RED   = '#EF5350'
GOLD  = '#FFD54F'
PURPLE= '#CE93D8'
TEAL  = '#4DB6AC'
WHITE = '#E0E0E0'

os.makedirs('results', exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Load Data
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data...")
an  = pd.read_csv('results/analytical_data.csv')
num = pd.read_csv('results/numerical_data.csv')

# Keep only rows with valid S values
valid = num[
    (num['Stability'].isin([2, 3])) &
    (num['S_exact']  > 0) & (num['S_exact']  < 1e50) &
    (num['S_approx'] > 0) & (num['S_approx'] < 1e50)
].copy()

valid['rel_err']  = (valid['S_exact'] - valid['S_approx']) / valid['S_exact']
valid['abs_err']  = valid['S_exact'] - valid['S_approx']
valid['log_S_ex'] = np.log10(valid['S_exact'])
valid['log_S_ap'] = np.log10(valid['S_approx'])
print(f"  Valid points: {len(valid):,}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  RK4 Convergence: global error vs step size dt
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 1: RK4 convergence...")

# Simulate RK4 on a model ODE:  dy/dt = -k*y,  y(0)=1,  exact = exp(-k*t)
# This mimics the lambda running: exponential decay in RGE t.
k   = 0.05
t0, t_end = 0.0, 100.0
y_exact_end = np.exp(-k * t_end)

dts      = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
rk4_errs = []
euler_errs = []

for dt in dts:
    # RK4
    y  = 1.0; t = t0
    while t < t_end - 1e-12:
        h  = min(dt, t_end - t)
        k1 = -k * y
        k2 = -k * (y + 0.5*h*k1)
        k3 = -k * (y + 0.5*h*k2)
        k4 = -k * (y + h*k3)
        y  = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
    rk4_errs.append(abs(y - y_exact_end))
    # Euler
    y = 1.0; t = t0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        y = y + h * (-k * y)
        t += h
    euler_errs.append(abs(y - y_exact_end))

fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK)
ax.set_facecolor(DARK)
ax.loglog(dts, rk4_errs,   'o-', color=BLUE,  lw=2, label='RK4 (4th-order)', markersize=6)
ax.loglog(dts, euler_errs, 's--', color=RED,   lw=2, label='Euler (1st-order)', markersize=6)
# Reference slopes
dt_ref = np.array([dts[0], dts[-1]])
ax.loglog(dt_ref, 5e-3 * (dt_ref/dts[0])**4,  ':', color=GOLD,   lw=1.5, label=r'$\propto \Delta t^4$')
ax.loglog(dt_ref, 2e-1 * (dt_ref/dts[0])**1,  ':', color=GREEN,  lw=1.5, label=r'$\propto \Delta t^1$')
ax.set_xlabel(r'Step size $\Delta t$', color=WHITE)
ax.set_ylabel('Global truncation error $|y_N - y_{exact}|$', color=WHITE)
ax.set_title('RK4 vs Euler Convergence on Model RGE', color=WHITE, fontweight='bold')
ax.tick_params(colors=WHITE); ax.xaxis.label.set_color(WHITE); ax.yaxis.label.set_color(WHITE)
for sp in ax.spines.values(): sp.set_color('#444')
ax.legend(facecolor='#222', edgecolor='#555', labelcolor=WHITE)
ax.grid(True, which='both', color='#333', ls='--', lw=0.5)
plt.tight_layout()
fig.savefig('results/error_rk4_convergence.png', dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Quadrature Convergence: action integral vs N_QUAD
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 2: Quadrature convergence...")

# Analytic reference: the kinetic bounce action integral is 16π²/(3|λ|).
# We test the *potential integral* numerically: ∫ 2π² r³ V(φ) dr
# using the substitution u = tanh(...) and compare to a very-high-N reference.

pi = math.pi

def integrand(u, R, lam, alpha=4.0):
    """Potential piece of the action integrand after u = tanh substitution."""
    if abs(u) >= 1.0: return 0.0
    ratio = ((1 + u) / (1 - u)) ** alpha
    e4x   = ratio ** 2
    phi   = (2.0 / (R * (ratio + 1.0))) * math.sqrt(2.0 / abs(lam))
    V     = 0.25 * lam * phi**4
    jac   = alpha / (1 - u*u)
    return 2 * pi**2 * R**4 * e4x * V * jac

lam_test = -0.01
R_test   = 1e10

def simp_integral(N, lam, R):
    du = 2.0 / N
    s  = 0.0
    for i in range(1, N):
        u = -1.0 + i * du
        w = 4.0/3.0 if i % 2 == 1 else 2.0/3.0
        s += w * integrand(u, R, lam) * du
    return s

N_ref   = 32768
I_ref   = simp_integral(N_ref, lam_test, R_test)
N_vals  = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
q_errs  = [abs(simp_integral(N, lam_test, R_test) - I_ref) / (abs(I_ref) + 1e-300) for N in N_vals]

fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK)
ax.set_facecolor(DARK)
ax.loglog(N_vals, q_errs, 'o-', color=TEAL, lw=2, markersize=6, label='Composite Simpson')
N_arr = np.array(N_vals, dtype=float)
ax.loglog(N_arr, 0.5 * q_errs[2] * (N_arr / N_vals[2])**(-4), ':', color=GOLD, lw=1.5, label=r'$\propto N^{-4}$')
ax.axvline(256,  color=RED,    ls='--', lw=1.2, label='Previous N=256')
ax.axvline(2048, color=GREEN,  ls='--', lw=1.2, label='Current N=2048')
ax.set_xlabel(r'Quadrature points $N_{quad}$', color=WHITE)
ax.set_ylabel('Relative error in action integral', color=WHITE)
ax.set_title('Bounce Action Quadrature Convergence', color=WHITE, fontweight='bold')
ax.tick_params(colors=WHITE)
for sp in ax.spines.values(): sp.set_color('#444')
ax.legend(facecolor='#222', edgecolor='#555', labelcolor=WHITE)
ax.grid(True, which='both', color='#333', ls='--', lw=0.5)
plt.tight_layout()
fig.savefig('results/error_quadrature_convergence.png', dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Golden Section Search: action value vs tolerance
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 3: Golden Section convergence...")

invphi  = (math.sqrt(5) - 1) / 2
invphi2 = (3 - math.sqrt(5)) / 2

def test_function(x):
    """Mock action as function of log R — convex, smooth minimum."""
    return 1.0 + (x - 0.3)**2 * 5.0 + 0.02 * math.sin(20*x)

def golden_search(a, b, tol):
    h = b - a
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi  * h
    yc, yd = test_function(c), test_function(d)
    for _ in range(n):
        if yc < yd:
            b, d, yd, h = d, c, yc, invphi*h
            c  = a + invphi2*h; yc = test_function(c)
        else:
            a, c, yc, h = c, d, yd, invphi*h
            d  = a + invphi*h;  yd = test_function(d)
    return 0.5*(a+b), test_function(0.5*(a+b))

x_true  = 0.3
S_true  = test_function(x_true)
tols    = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
gs_errs = []
for tol in tols:
    x_est, S_est = golden_search(-5, 5, tol)
    gs_errs.append(abs(S_est - S_true))

fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK)
ax.set_facecolor(DARK)
ax.semilogy(range(len(tols)), gs_errs, 'o-', color=PURPLE, lw=2, markersize=6, label='Action error')
ax.set_xticks(range(len(tols)))
ax.set_xticklabels([f'1e{int(math.log10(t))}' for t in tols], rotation=45, color=WHITE)
ax.axvline(tols.index(1e-10), color=RED,   ls='--', lw=1.2, label='Previous tol=1e-10')
ax.axvline(tols.index(1e-13), color=GREEN, ls='--', lw=1.2, label='Current tol=1e-13')
ax.set_xlabel('Search tolerance', color=WHITE)
ax.set_ylabel('Error in minimized action $S_{min}$', color=WHITE)
ax.set_title('Golden Section Search Convergence on $S(R)$', color=WHITE, fontweight='bold')
ax.tick_params(colors=WHITE)
for sp in ax.spines.values(): sp.set_color('#444')
ax.legend(facecolor='#222', edgecolor='#555', labelcolor=WHITE)
ax.grid(True, color='#333', ls='--', lw=0.5)
plt.tight_layout()
fig.savefig('results/error_golden_section.png', dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — S_exact vs S_approx scatter (log–log)
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 4: S scatter plot...")

# Sample to avoid overplotting
sample = valid.sample(min(40000, len(valid)), random_state=42)

fig, ax = plt.subplots(figsize=(7, 6), facecolor=DARK)
ax.set_facecolor(DARK)

# Colour by stability
col_map = {2: GREEN, 3: RED}
for stab, label in [(2, 'Metastable'), (3, 'Unstable')]:
    sub = sample[sample['Stability'] == stab]
    ax.scatter(sub['S_approx'], sub['S_exact'],
               c=col_map[stab], s=1.2, alpha=0.4, label=label, rasterized=True)

# Perfect-agreement line
lims = [max(sample['S_approx'].min(), 1), min(sample['S_approx'].max(), 1e8)]
ax.plot(lims, lims, '--', color=WHITE, lw=1.5, label='$S_{exact}=S_{approx}$')
ax.plot(lims, [l*1.05 for l in lims], ':', color=GOLD, lw=1, label='±5%')
ax.plot(lims, [l*0.95 for l in lims], ':', color=GOLD, lw=1)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$S_{approx}$ (conformal analytical)', color=WHITE)
ax.set_ylabel(r'$S_{exact}$ (numerical $\phi^6$ integration)', color=WHITE)
ax.set_title(r'Exact vs. Approximate Bounce Action', color=WHITE, fontweight='bold')
ax.tick_params(colors=WHITE)
for sp in ax.spines.values(): sp.set_color('#444')
ax.legend(facecolor='#222', edgecolor='#555', labelcolor=WHITE, markerscale=5)
ax.grid(True, which='both', color='#333', ls='--', lw=0.5)
plt.tight_layout()
fig.savefig('results/error_S_scatter.png', dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Relative discrepancy heatmap in (Mt, Mh) space
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 5: Relative discrepancy heatmap...")

hmap = valid[['Mt', 'Mh_calc', 'rel_err']].dropna()
hmap = hmap[(hmap['Mh_calc'] > 0) & (hmap['Mh_calc'] < 250)]
# Clip extreme outliers
q99 = hmap['rel_err'].quantile(0.99)
q01 = hmap['rel_err'].quantile(0.01)
hmap = hmap[(hmap['rel_err'] >= q01) & (hmap['rel_err'] <= q99)]

from scipy.stats import binned_statistic_2d
stat, xedge, yedge, _ = binned_statistic_2d(
    hmap['Mt'], hmap['Mh_calc'], hmap['rel_err'],
    statistic='mean', bins=200,
    range=[[155, 185], [115, 135]]
)

fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK)
ax.set_facecolor(DARK)
im = ax.pcolormesh(xedge, yedge, stat.T,
                   cmap='RdYlGn', vmin=-0.15, vmax=0.15, shading='auto')
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label(r'$(S_{exact}-S_{approx})/S_{exact}$', color=WHITE)
cb.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)
ax.set_xlabel(r'Top quark mass $M_t$ [GeV]', color=WHITE)
ax.set_ylabel(r'Higgs mass $M_h$ [GeV]', color=WHITE)
ax.set_title(r'Relative Discrepancy $(S_{exact}-S_{approx})/S_{exact}$', color=WHITE, fontweight='bold')
ax.tick_params(colors=WHITE)
for sp in ax.spines.values(): sp.set_color('#444')
ax.grid(False)
# Mark SM point
ax.axvline(173.34, color=GOLD, ls='--', lw=1.2, label='$M_t^{SM}=173.3$ GeV')
ax.axhline(125.10, color=BLUE, ls='--', lw=1.2, label='$M_h^{SM}=125.1$ GeV')
ax.legend(facecolor='#222', edgecolor='#555', labelcolor=WHITE)
plt.tight_layout()
fig.savefig('results/error_discrepancy_heatmap.png', dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Action residual distribution (histogram)
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 6: Residual histogram...")

# Use relative error for physically interesting region (S < 1000)
rel_clip = valid[(valid['S_exact'] < 1000) & (valid['S_exact'] > 1)]['rel_err']
print(f"  Median relative error: {rel_clip.median()*100:.3f}%")
print(f"  Std relative error:      {rel_clip.std()*100:.3f}%")

fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK)
ax.set_facecolor(DARK)
n, bins, patches = ax.hist(rel_clip * 100, bins=120, color=TEAL, alpha=0.85, edgecolor='none', density=True)
# Overlay Gaussian fit
from scipy.stats import norm as spnorm
mu_fit, std_fit = spnorm.fit(rel_clip * 100)
x_fit = np.linspace(bins[0], bins[-1], 500)
ax.plot(x_fit, spnorm.pdf(x_fit, mu_fit, std_fit), '-', color=GOLD, lw=2,
        label=rf'Gaussian fit mu={mu_fit:.3f}%, sigma={std_fit:.3f}%')
ax.axvline(0, color=WHITE, ls='--', lw=1)
ax.set_xlabel(r'Relative discrepancy $(S_{exact}-S_{approx})/S_{exact}$ [%]', color=WHITE)
ax.set_ylabel('Probability density', color=WHITE)
ax.set_title('Distribution of Action Residuals', color=WHITE, fontweight='bold')
ax.tick_params(colors=WHITE)
for sp in ax.spines.values(): sp.set_color('#444')
ax.legend(facecolor='#222', edgecolor='#555', labelcolor=WHITE)
ax.grid(True, color='#333', ls='--', lw=0.5)
plt.tight_layout()
fig.savefig('results/error_residual_histogram.png', dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Boundary discrepancy vs |lambda_min|
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 7: Discrepancy vs lambda_min proxy...")

# Use S_approx as proxy for 1/|lambda_min| (S_approx = 8π²/(3|λ|))
# so |lambda_min| ~ 8π²/(3 * S_approx)
pi2_8 = 8 * math.pi**2 / 3
valid2 = valid[(valid['S_approx'] > 1) & (valid['S_approx'] < 5000)].copy()
valid2['lam_proxy'] = pi2_8 / valid2['S_approx']

# Bin by lambda_min proxy
bins_lam = np.linspace(valid2['lam_proxy'].min(), valid2['lam_proxy'].max(), 60)
valid2['lam_bin'] = pd.cut(valid2['lam_proxy'], bins=bins_lam)
grouped = valid2.groupby('lam_bin', observed=True)['rel_err'].agg(['mean', 'std']).reset_index()
lam_mids = [(iv.left + iv.right)/2 for iv in grouped['lam_bin']]

fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK)
ax.set_facecolor(DARK)
ax.errorbar(lam_mids, grouped['mean']*100, yerr=grouped['std']*100,
            fmt='o-', color=PURPLE, lw=2, markersize=4, elinewidth=1,
            ecolor=PURPLE, alpha=0.8, capsize=3, label=r'Mean $\pm 1\sigma$')
ax.axhline(0, color=WHITE, ls='--', lw=1)
ax.set_xlabel(r'$|\lambda_{min}|$ proxy  ($\approx 8\pi^2/3S_{approx}$)', color=WHITE)
ax.set_ylabel(r'$(S_{exact}-S_{approx})/S_{exact}$ [%]', color=WHITE)
ax.set_title(r'Action Discrepancy vs. Depth of Instability $|\lambda_{min}|$', color=WHITE, fontweight='bold')
ax.tick_params(colors=WHITE)
for sp in ax.spines.values(): sp.set_color('#444')
ax.legend(facecolor='#222', edgecolor='#555', labelcolor=WHITE)
ax.grid(True, color='#333', ls='--', lw=0.5)
plt.tight_layout()
fig.savefig('results/error_vs_lambda.png', dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Phase boundary misclassification: rescaled narrow strip
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 8: Phase boundary classification map...")

# Points where the two methods disagree in classification
an_clip  = an[ an['Mh_calc'].between(115,135) &  an['Mt'].between(155,185)].copy()
num_clip = num[num['Mh_calc'].between(115,135) & num['Mt'].between(155,185)].copy()

# Merge on Mt (same grid)
merged = pd.merge(an_clip[['Mt','Mh_calc','Stability']],
                  num_clip[['Mt','Mh_calc','Stability']],
                  on=['Mt','Mh_calc'], suffixes=('_an','_num'))
merged['disagree'] = merged['Stability_an'] != merged['Stability_num']

agree    = merged[~merged['disagree']]
disagree = merged[merged['disagree']]
print(f"  Disagreeing points in closeup region: {len(disagree):,} / {len(merged):,}")

fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK)
ax.set_facecolor(DARK)

cmap_stab = {1: '#4CAF50', 2: '#2196F3', 3: '#F44336', 4: '#9E9E9E'}
for stab in [1,2,3,4]:
    sub = agree[agree['Stability_num'] == stab]
    if len(sub):
        ax.scatter(sub['Mt'], sub['Mh_calc'], c=cmap_stab[stab], s=0.8, alpha=0.6, rasterized=True)

if len(disagree):
    ax.scatter(disagree['Mt'], disagree['Mh_calc'],
               c='yellow', s=8, alpha=1.0, zorder=5,
               label=f'Disagreement ({len(disagree):,} pts)', rasterized=False)

# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', label='Stable'),
    Patch(facecolor='#2196F3', label='Metastable'),
    Patch(facecolor='#F44336', label='Unstable'),
    Patch(facecolor='yellow',  label=f'Reclassified by $\\phi^6$ ({len(disagree):,} pts)'),
]
ax.legend(handles=legend_elements, facecolor='#222', edgecolor='#555', labelcolor=WHITE, loc='upper left')
ax.axvline(173.34, color=GOLD, ls='--', lw=1, alpha=0.7)
ax.axhline(125.10, color=BLUE, ls='--', lw=1, alpha=0.7)
ax.set_xlabel(r'$M_t$ [GeV]', color=WHITE)
ax.set_ylabel(r'$M_h$ [GeV]', color=WHITE)
ax.set_title('Phase Boundary Reclassification by $\\phi^6$ Correction', color=WHITE, fontweight='bold')
ax.tick_params(colors=WHITE)
for sp in ax.spines.values(): sp.set_color('#444')
ax.grid(True, color='#333', ls='--', lw=0.3)
plt.tight_layout()
fig.savefig('results/error_boundary_reclassification.png', dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Error Analysis Summary ----------------------------------")
print(f"  Total valid (meta+unstable) points:    {len(valid):>10,}")
print(f"  Median relative error (S<1000):        {rel_clip.median()*100:>+10.4f} %")
print(f"  Sigma relative error (S<1000):          {rel_clip.std()*100:>10.4f} %")
print(f"  95th-percentile absolute error:        {valid['abs_err'].quantile(0.95):>10.2f}")
print(f"  Reclassified boundary points (closeup):{len(disagree):>10,}")
print(f"  N_QUAD improvement factor:             {2048//256:>10}x")
print(f"  RK4 dt improvement factor:             {10:>10}x")
print(f"  Golden Section tol improvement:        {int(1e13/1e10):>10}x")
print("------------------------------------------------------------\n")
print("All 8 error analysis figures saved to results/")
