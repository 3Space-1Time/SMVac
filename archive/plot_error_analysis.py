"""
plot_error_analysis.py
======================
Publication-ready error analysis comparing the analytical bounce-action
approximation  S_approx = 8pi^2/(3|lam_eff|)  against the exact SimpleBounce
numerical result S_exact, over the dense 770 000-point vacuum-stability grid.

Outputs (all written to results/):
  1. error_discrepancy_heatmap.png        - percentage-error heat map in (Mh, Mt)
  2. error_S_scatter.png                  - three-panel S_exact vs S_approx log scatter
  3. error_residual_histogram.png         - distribution of relative errors
  4. error_boundary_reclassification.png  - where the approximation mis-classifies
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR  = "results"
CSV_PATH = os.path.join(OUT_DIR, "numerical_dense_closeup.csv")
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    sys.exit(f"[ERROR] Data file not found: {CSV_PATH}")

# ── global light publication style ────────────────────────────────────────────
BG_COLOR  = "#ffffff"
TEXT_COL  = "#000000"
GRID_COL  = "#e0e0e0"
ACCENT    = "#1f77b4"
GOLD      = "#ff7f0e"
SM_COLOR  = "#d62728"
SM_MH, SM_MT = 125.1, 173.1
S_THRESHOLD_NOMINAL = 450.0

mpl.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "axes.edgecolor": "#333333", "axes.labelcolor": TEXT_COL,
    "axes.titlecolor": TEXT_COL, "axes.titlesize": 13,
    "axes.labelsize": 11, "axes.titlepad": 10,
    "xtick.color": TEXT_COL, "ytick.color": TEXT_COL,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID_COL, "grid.linewidth": 0.5, "grid.alpha": 0.8,
    "legend.facecolor": "#f8f9fa", "legend.edgecolor": "#cccccc",
    "legend.labelcolor": TEXT_COL, "legend.fontsize": 9,
    "text.color": TEXT_COL, "font.family": "serif",
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": BG_COLOR,
})

def sm_star(ax, zorder=10, size=180):
    ax.scatter(SM_MH, SM_MT, marker="*", s=size, color=SM_COLOR,
               zorder=zorder, linewidths=0.6, edgecolors="black",
               label=fr"SM  ($M_h={SM_MH}$, $M_t={SM_MT}$)")

def fmt_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title, pad=12)
    ax.grid(True, which="both", linestyle="--")
    ax.tick_params(which="both", direction="in", length=4, width=0.6)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path); plt.close(fig)
    print(f"  Saved -> {path}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading data ...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
required = {"mt", "mh_calc", "stability", "s_exact", "s_approx"}
missing  = required - set(df.columns)
if missing:
    sys.exit(f"[ERROR] Missing columns: {missing}  |  Found: {list(df.columns)}")
df.rename(columns={"mh_calc": "mh"}, inplace=True)

print(f"  Total rows            : {len(df):,}")
print(f"  Stability code counts : {df['stability'].value_counts().sort_index().to_dict()}")

mask_valid = (df["s_exact"] > 0) & (df["s_approx"] > 0) & \
             np.isfinite(df["s_exact"]) & np.isfinite(df["s_approx"])
dv = df[mask_valid].copy()
dv["rel_err"] = (dv["s_exact"] - dv["s_approx"]) / dv["s_exact"] * 100.0
dv["abs_err"] = dv["rel_err"].abs()

print(f"  Points with valid S   : {len(dv):,}")
print(f"  rel_err range         : [{dv['rel_err'].min():.2f}%,  {dv['rel_err'].max():.2f}%]")

# ═════════════════════════════════════════════════════════════════════════════
# 2. PLOT 1 – HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Plotting error heatmap ...")
fig, ax = plt.subplots(figsize=(9, 7))
clamp = 20.0
plot_err = dv["rel_err"].clip(-clamp, clamp)
cmap = mpl.colormaps.get_cmap("RdYlBu_r").copy()
cmap.set_under("#000080"); cmap.set_over("#800000")
levels = np.linspace(-clamp, clamp, 41)
cf = ax.tricontourf(dv["mh"], dv["mt"], plot_err,
                    levels=levels, cmap=cmap, extend="both")
cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.035)
cb.set_label(r"$(S_\mathrm{exact}-S_\mathrm{approx})/S_\mathrm{exact}$ [%]", fontsize=10)
cb.ax.yaxis.set_tick_params(color=TEXT_COL, labelcolor=TEXT_COL)
cb.outline.set_edgecolor("#333333")
ax.axvline(SM_MH, color="black", lw=0.8, ls="--", alpha=0.7,
           label=fr"$M_h^\mathrm{{SM}}={SM_MH}$ GeV")
ax.axhline(SM_MT, color="black", lw=0.8, ls="--", alpha=0.7,
           label=fr"$M_t^\mathrm{{SM}}={SM_MT}$ GeV")
sm_star(ax)
fmt_ax(ax,
       xlabel=r"Higgs mass $M_h$ [GeV]",
       ylabel=r"Top quark mass $M_t$ [GeV]",
       title=r"Relative Discrepancy $(S_\mathrm{exact}-S_\mathrm{approx})/S_\mathrm{exact}$")
ax.legend(loc="upper left", framealpha=0.9)
save(fig, "error_discrepancy_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PLOT 2 – THREE-PANEL SCATTER
# ═════════════════════════════════════════════════════════════════════════════
print("[3/5] Plotting S scatter ...")
fig, axes = plt.subplots(1, 3, figsize=(17, 6),
                          gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]})
fig.suptitle(
    r"Bounce Action Comparison: Exact Numerical ($\phi^6$) vs Conformal Analytical Approximation",
    fontsize=13, y=1.01)

meta_mask = dv["stability"] == 2
unst_mask = dv["stability"] == 3

# ── (a) full log scatter ──────────────────────────────────────────────────────
ax = axes[0]
ax.set_title("(a)  Full Scatter — Physically Relevant Range", fontsize=10)
for mask, col, lab in [(meta_mask, "#2ca02c", "Metastable"),
                       (unst_mask, "#d62728", "Unstable")]:
    ax.scatter(dv.loc[mask, "s_approx"], dv.loc[mask, "s_exact"],
               s=0.8, alpha=0.35, color=col, rasterized=True, label=lab)
smin = max(dv[["s_exact","s_approx"]].min().min(), 1e-1)
smax = dv[["s_exact","s_approx"]].max().max() * 1.05
lims = [smin, smax]
ax.plot(lims, lims, "--", color="black", lw=1.0, alpha=0.7,
        label=r"$S_\mathrm{exact}=S_\mathrm{approx}$")
band_x = np.array(lims)
for pct, col, ls in [(0.05, GOLD, ":"), (0.15, "#9467bd", ":")]:
    ax.fill_between(band_x, band_x*(1-pct), band_x*(1+pct), alpha=0.1, color=col)
    ax.plot(band_x, band_x*(1+pct), ls=ls, lw=1.0, color=col, alpha=0.8,
            label=f"±{int(pct*100)}%")
    ax.plot(band_x, band_x*(1-pct), ls=ls, lw=1.0, color=col, alpha=0.8)
frac5  = (dv["abs_err"] < 5).mean()
frac15 = (dv["abs_err"] < 15).mean()
ax.text(0.04, 0.94,
        f"{frac5*100:.1f}% within ±5%\n{frac15*100:.1f}% within ±15%",
        transform=ax.transAxes, fontsize=8.5, color="#000000", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lims); ax.set_ylim(lims)
fmt_ax(ax, xlabel=r"$S_\mathrm{approx}$",
       ylabel=r"$S_\mathrm{exact}$  (numerical $\phi^6$ integration)")
ax.legend(fontsize=8, markerscale=5, loc="upper left")

# ── (b) relative discrepancy vs S_approx ─────────────────────────────────────
ax = axes[1]
ax.set_title(r"(b)  Relative Discrepancy vs Action Value", fontsize=10)
s_vals  = dv["s_approx"].values
re_vals = dv["rel_err"].values
log_edges = np.logspace(np.log10(max(s_vals.min(),1)), np.log10(s_vals.max()), 60)
bc, bm, b16, b84 = [], [], [], []
for lo, hi in zip(log_edges[:-1], log_edges[1:]):
    m = (s_vals >= lo) & (s_vals < hi)
    if m.sum() < 5: continue
    v = re_vals[m]
    bc.append(np.sqrt(lo*hi)); bm.append(np.mean(v))
    b16.append(np.percentile(v,16)); b84.append(np.percentile(v,84))
bc, bm, b16, b84 = np.array(bc), np.array(bm), np.array(b16), np.array(b84)
ax.scatter(dv["s_approx"], dv["rel_err"].clip(-35,35),
           s=0.5, alpha=0.1, color="#888888", rasterized=True)
for pct, col, ls, lab in [(5, GOLD, ":", "±5%"), (15, "#9467bd", ":", "±15%")]:
    ax.axhline( pct, color=col, lw=1.0, ls=ls, alpha=0.8, label=lab)
    ax.axhline(-pct, color=col, lw=1.0, ls=ls, alpha=0.8)
ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
if len(bc):
    ax.fill_between(bc, b16.clip(-35), b84.clip(35), alpha=0.22, color=ACCENT, label="±1σ")
    ax.plot(bc, bm, color=ACCENT, lw=1.5, label="Binned mean")
ax.text(0.05, 0.90,
        r"Scatter grows" "\n" r"near threshold" "\n"
        r"($S$ small $\to$ large $|\lambda_\mathrm{min}|$ small)",
        transform=ax.transAxes, fontsize=7.5, color="#d62728",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
ax.text(0.55, 0.10,
        r"Approx." "\n" r"accurate" "\n" r"($|\lambda_\mathrm{min}|$ small)",
        transform=ax.transAxes, fontsize=7.5, color="#2ca02c",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
ax.set_xscale("log"); ax.set_ylim(-35, 35)
fmt_ax(ax, xlabel=r"$S_\mathrm{approx}$",
       ylabel=r"$(S_\mathrm{exact}-S_\mathrm{approx})/S_\mathrm{exact}$ [%]")
ax.legend(fontsize=8, loc="upper right")

# ── (c) zoomed near threshold ─────────────────────────────────────────────────
ax = axes[2]
ax.set_title(r"(c)  Zoomed: Near Metastability Threshold", fontsize=10)
zoom_max = S_THRESHOLD_NOMINAL * 4.5
mask_z = (dv["s_exact"] < zoom_max) & (dv["s_approx"] < zoom_max)
dz = dv[mask_z]
for mask, col, lab in [(dz["stability"]==2, "#2ca02c","Metastable"),
                       (dz["stability"]==3, "#d62728","Unstable")]:
    ax.scatter(dz.loc[mask,"s_approx"], dz.loc[mask,"s_exact"],
               s=1.5, alpha=0.5, color=col, rasterized=True, label=lab)
ax.plot([0,zoom_max],[0,zoom_max],"--",color="black",lw=1.0,alpha=0.7)
ax.axvline(S_THRESHOLD_NOMINAL, color="#1f77b4", lw=1.2, ls="-.", alpha=0.8)
ax.axhline(S_THRESHOLD_NOMINAL, color="#1f77b4", lw=1.2, ls="-.", alpha=0.8)
ax.set_xlim(0, zoom_max); ax.set_ylim(0, zoom_max)
ax.text(0.05, 0.60,
        r"$S_\mathrm{approx}<S_\mathrm{thr}$ but" + "\n" +
        r"$S_\mathrm{exact}>S_\mathrm{thr}$" + "\n→ Numerical rescues\n    to Metastable",
        transform=ax.transAxes, fontsize=7.5, color="#2ca02c",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
ax.text(0.55, 0.08,
        r"$S_\mathrm{approx}>S_\mathrm{thr}$ but" + "\n" +
        r"$S_\mathrm{exact}<S_\mathrm{thr}$" + "\n→ Numerical downgrades\n    to Unstable",
        transform=ax.transAxes, fontsize=7.5, color="#d62728",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
ax.text(0.52, 0.83,
        f"Typical $S_\\mathrm{{threshold}}\\approx{S_THRESHOLD_NOMINAL:.0f}$",
        transform=ax.transAxes, fontsize=7.5, color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9))
fmt_ax(ax, xlabel=r"$S_\mathrm{approx}$", ylabel=r"$S_\mathrm{exact}$")
ax.legend(fontsize=8, markerscale=5, loc="upper left")
ax.text(0.37, 0.96, f"· dash-dot = $S_\\mathrm{{threshold}} \\approx {S_THRESHOLD_NOMINAL:.0f}$",
        transform=ax.transAxes, fontsize=7, color="#1f77b4", alpha=0.9)

fig.tight_layout(rect=[0,0,1,0.98])
save(fig, "error_S_scatter.png")

# ═════════════════════════════════════════════════════════════════════════════
# 4. PLOT 3 – RESIDUAL HISTOGRAM
# ═════════════════════════════════════════════════════════════════════════════
print("[4/5] Plotting residual histogram ...")
rel_err = dv["rel_err"].values
fig, ax = plt.subplots(figsize=(10, 6))
clip_lo, clip_hi = np.percentile(rel_err, [0.5, 99.5])
data_c = rel_err.clip(clip_lo, clip_hi)
counts, edges, patches = ax.hist(data_c, bins=120, density=True,
                                  color=ACCENT, alpha=0.6, edgecolor="white", linewidth=0.5,
                                  label="All valid points")
mid_pts = 0.5*(edges[:-1]+edges[1:])
for patch, mid in zip(patches, mid_pts):
    if mid < 0:
        patch.set_facecolor("#d62728"); patch.set_alpha(0.7)
try:
    kde_x = np.linspace(clip_lo, clip_hi, 600)
    kde = gaussian_kde(rel_err, bw_method="scott")
    ax.plot(kde_x, kde(kde_x), color="black", lw=1.5, alpha=0.85, label="KDE")
except Exception:
    pass
mu   = np.mean(rel_err);    med = np.median(rel_err)
p75  = np.percentile(rel_err, 75); p95 = np.percentile(rel_err, 95)
p99  = np.percentile(rel_err, 99)
frac_pos = (rel_err > 0).mean()*100
stats_text = (f"N = {len(rel_err):,}\n"
              f"Mean   = {mu:.2f}%\n"
              f"Median = {med:.2f}%\n"
              f"P75    = {p75:.2f}%\n"
              f"P95    = {p95:.2f}%\n"
              f"P99    = {p99:.2f}%\n"
              f"S_exact > S_approx : {frac_pos:.1f}%")
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", family="monospace", color=TEXT_COL,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff",
                  edgecolor="#cccccc", alpha=0.95))
for val, col, ls, lw, lab in [
    (mu,  "#ff7f0e", "-",  1.4, f"Mean ({mu:.2f}%)"),
    (med, "#1f77b4", "--", 1.2, f"Median ({med:.2f}%)"),
    (0,   "black",   "-",  1.0, "Zero (no error)"),
]:
    ax.axvline(val, color=col, lw=lw, ls=ls, alpha=0.85, label=lab)
ylim = ax.get_ylim()[1]
ax.axvspan(clip_lo, 0, alpha=0.1, color="#d62728")
ax.axvspan(0, clip_hi, alpha=0.1, color="#2ca02c")
ax.text(0.02, 0.5, "S_exact < S_approx\n(approx over-estimates)",
        ha="left", va="center", fontsize=8, color="#d62728",
        transform=ax.transAxes)
ax.text(0.58, 0.5, "S_exact > S_approx\n(approx under-estimates)",
        ha="left", va="center", fontsize=8, color="#2ca02c",
        transform=ax.transAxes)
fmt_ax(ax,
       xlabel=r"$(S_\mathrm{exact}-S_\mathrm{approx})/S_\mathrm{exact}$ [%]",
       ylabel="Probability density",
       title="Distribution of Action Residuals")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
save(fig, "error_residual_histogram.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5. PLOT 4 – BOUNDARY RECLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════
print("[5/5] Plotting boundary reclassification ...")
Sthr = S_THRESHOLD_NOMINAL
dv["class_approx"] = np.where(dv["s_approx"] >= Sthr, 2, 3)
mask_rescued = ((dv["stability"] == 2) & (dv["class_approx"] == 3) &
                (dv["s_exact"] >= Sthr) & (dv["s_approx"] < Sthr))
mask_demoted = ((dv["stability"] == 3) & (dv["class_approx"] == 2) &
                (dv["s_exact"] < Sthr)  & (dv["s_approx"] >= Sthr))
n_rescued = mask_rescued.sum(); n_demoted = mask_demoted.sum()
print(f"  Rescued  (Unstable->Metastable by exact): {n_rescued:,}")
print(f"  Demoted  (Metastable->Unstable by exact): {n_demoted:,}")

fig, ax = plt.subplots(figsize=(10, 8))
stab_cmap = {1: "#d9f0d3", 2: "#a6dba0", 3: "#f4a582", 4: "#d9d9d9"}
for code, col in stab_cmap.items():
    sub = df[df["stability"] == code]
    if len(sub):
        ax.scatter(sub["mh"], sub["mt"], s=0.4, color=col, alpha=0.7, rasterized=True)
RESCUED_COL = "#1f77b4"; DEMOTED_COL = "#d62728"
if n_rescued > 0:
    dr = dv[mask_rescued]
    ax.scatter(dr["mh"], dr["mt"], s=12, color=RESCUED_COL, alpha=0.9, zorder=5,
               rasterized=True, label=f"Rescued: Unstable→Metastable ({n_rescued:,} pts)")
if n_demoted > 0:
    dd = dv[mask_demoted]
    ax.scatter(dd["mh"], dd["mt"], s=12, color=DEMOTED_COL, alpha=0.9, zorder=5,
               rasterized=True, label=f"Demoted: Metastable→Unstable ({n_demoted:,} pts)")
ax.axvline(SM_MH, color="black", lw=0.9, ls="--", alpha=0.8,
           label=fr"$M_h^\mathrm{{SM}}={SM_MH}$ GeV")
ax.axhline(SM_MT, color="black", lw=0.9, ls="--", alpha=0.8,
           label=fr"$M_t^\mathrm{{SM}}={SM_MT}$ GeV")
sm_star(ax, size=220)
legend_patches = [Patch(color="#d9f0d3", label="Stable (1)"),
                  Patch(color="#a6dba0", label="Metastable (2)"),
                  Patch(color="#f4a582", label="Unstable (3)"),
                  Patch(color="#d9d9d9", label="Non-perturbative (4)")]
leg1 = ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5,
                 title="Background (exact solver)", title_fontsize=8.5, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
fmt_ax(ax,
       xlabel=r"Higgs mass $M_h$ [GeV]",
       ylabel=r"Top quark mass $M_t$ [GeV]",
       title=r"Phase Boundary Reclassification by $\phi^6$ Correction")
fig.tight_layout()
save(fig, "error_boundary_reclassification.png")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  Total grid points           : {len(df):,}")
print(f"  Points with valid S values  : {len(dv):,}")
print(f"  Mean   rel. error           : {dv['rel_err'].mean():.3f}%")
print(f"  Median rel. error           : {dv['rel_err'].median():.3f}%")
print(f"  Fraction within +/-5%       : {(dv['abs_err'] < 5).mean()*100:.1f}%")
print(f"  Fraction within +/-15%      : {(dv['abs_err'] < 15).mean()*100:.1f}%")
print(f"  Fraction S_exact > S_approx : {(dv['rel_err'] > 0).mean()*100:.1f}%")
print(f"  Reclassified rescued        : {n_rescued:,}")
print(f"  Reclassified demoted        : {n_demoted:,}")
print(f"\n  All plots saved to {OUT_DIR}/")
print("="*60)
