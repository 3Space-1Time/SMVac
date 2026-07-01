import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/numerical_data.csv')

# ---------------  PANEL A: Phase diagram with sandwich highlighted  ---------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'height_ratios': [1, 1]})
fig.suptitle('Sandwiched Unstable Region: Analysis of the 2→3→2 Pattern', fontsize=14, fontweight='bold')

# Panel 1: Full stability diagram in (Mh, Mt) plane
ax = axes[0, 0]
sc = ax.scatter(df['Mh_calc'], df['Mt'], c=df['Stability'], 
                cmap=ListedColormap(['green', 'yellow', 'red', 'black']),
                norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4),
                s=1, alpha=0.6)
ax.set_xlabel(r'$M_h$ [GeV]')
ax.set_ylabel(r'$M_t$ [GeV]')
ax.set_title('(a) Full Phase Diagram')
cbar = plt.colorbar(sc, ax=ax, ticks=[1, 2, 3, 4])
cbar.set_label('Stability')
cbar.ax.set_yticklabels(['Stable(1)', 'Metastable(2)', 'Unstable(3)', 'Non-pert(4)'])

# Panel 2: S_exact profile at Mh=100 GeV showing the 2→3→2→3 pattern
ax = axes[0, 1]
mh_target, tol = 100.0, 0.02
chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt').copy()
mask_pos = chunk['S_exact'] > 0
mask_neg = chunk['S_exact'] < 0
mask_sent = chunk['S_exact'] == -1.0
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.scatter(chunk.loc[mask_pos, 'Mt'], chunk.loc[mask_pos, 'S_exact'], 
           c='blue', s=8, label=r'$S_{\rm exact} > 0$', alpha=0.7)
ax.scatter(chunk.loc[mask_neg, 'Mt'], chunk.loc[mask_neg, 'S_exact'], 
           c='red', s=8, label=r'$S_{\rm exact} < 0$', alpha=0.7)
if mask_sent.any():
    ax.scatter(chunk.loc[mask_sent, 'Mt'], chunk.loc[mask_sent, 'S_exact'], 
               c='gray', s=8, marker='x', label='Sentinel (-1)', alpha=0.5)
# Add stability shading
for _, row in chunk.iterrows():
    if row['Stability'] == 2:
        ax.axvspan(row['Mt']-0.125, row['Mt']+0.125, alpha=0.1, color='yellow')
    elif row['Stability'] == 3:
        ax.axvspan(row['Mt']-0.125, row['Mt']+0.125, alpha=0.1, color='red')
ax.set_yscale('symlog', linthresh=100)
ax.set_xlabel(r'$M_t$ [GeV]')
ax.set_ylabel(r'$S_{\rm exact}$')
ax.set_title(rf'(b) Action at $M_h = {mh_target}$ GeV (symlog)')
ax.legend(fontsize=8)
ax.set_ylim(-1e7, 1e5)
ax.set_xlim(150, 235)

# Panel 3: S_exact profile at Mh=50 GeV with sandwich
ax = axes[0, 2]
mh_target, tol = 50.0, 0.02
chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt').copy()
mask_pos = chunk['S_exact'] > 0
mask_neg = chunk['S_exact'] < 0
mask_sent = chunk['S_exact'] == -1.0
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.scatter(chunk.loc[mask_pos, 'Mt'], chunk.loc[mask_pos, 'S_exact'], c='blue', s=8, alpha=0.7)
ax.scatter(chunk.loc[mask_neg, 'Mt'], chunk.loc[mask_neg, 'S_exact'], c='red', s=8, alpha=0.7)
for _, row in chunk.iterrows():
    if row['Stability'] == 2:
        ax.axvspan(row['Mt']-0.125, row['Mt']+0.125, alpha=0.1, color='yellow')
    elif row['Stability'] == 3:
        ax.axvspan(row['Mt']-0.125, row['Mt']+0.125, alpha=0.1, color='red')
ax.set_yscale('symlog', linthresh=100)
ax.set_xlabel(r'$M_t$ [GeV]')
ax.set_ylabel(r'$S_{\rm exact}$')
ax.set_title(rf'(c) Action at $M_h = {mh_target}$ GeV (symlog)')
ax.set_ylim(-1e7, 1e5)
ax.set_xlim(120, 230)

# Panel 4: Sandwiched region width vs M_h
ax = axes[1, 0]
mh_vals = np.arange(0, 100, 0.25)
lower_2_widths = []
upper_2_widths = []
sw_widths = []
mh_used = []
for mh_target in mh_vals:
    tol = 0.02
    chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt')
    if len(chunk) < 10: continue
    stabs = chunk['Stability'].values.astype(int)
    mt_vals = chunk['Mt'].values
    blocks = []
    current_s = stabs[0]; current_start = mt_vals[0]
    for i in range(1, len(stabs)):
        if stabs[i] != current_s:
            blocks.append((int(current_s), current_start, mt_vals[i-1]))
            current_s = stabs[i]; current_start = mt_vals[i]
    blocks.append((int(current_s), current_start, mt_vals[-1]))
    for i in range(1, len(blocks)-1):
        if blocks[i-1][0] == 2 and blocks[i][0] == 3 and blocks[i+1][0] == 2:
            if blocks[i][2] > blocks[i-1][1]:  # valid sandwich
                lower_2_widths.append(blocks[i-1][2] - blocks[i-1][1])
                upper_2_widths.append(blocks[i+1][2] - blocks[i+1][1])
                sw_widths.append(blocks[i][2] - blocks[i][1])
                mh_used.append(mh_target)
            break
ax.plot(mh_used, sw_widths, 'r-', lw=1.5, label='Sandwiched 3 width')
ax.plot(mh_used, upper_2_widths, 'g-', lw=1.5, label='Upper 2 band width')
ax.fill_between(mh_used, 0, sw_widths, color='red', alpha=0.15)
ax.set_xlabel(r'$M_h$ [GeV]')
ax.set_ylabel(r'Width in $\Delta M_t$ [GeV]')
ax.set_title('(d) Band Widths vs Higgs Mass')
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.set_ylim(0.05, 100)

# Panel 5: Lower and upper flank boundaries vs M_h  
ax = axes[1, 1]
lower_starts = []; lower_ends = []; upper_starts = []; upper_ends = []
mh_used2 = []
for mh_target in mh_vals:
    tol = 0.02
    chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt')
    if len(chunk) < 10: continue
    stabs = chunk['Stability'].values.astype(int)
    mt_vals = chunk['Mt'].values
    blocks = []
    current_s = stabs[0]; current_start = mt_vals[0]
    for i in range(1, len(stabs)):
        if stabs[i] != current_s:
            blocks.append((int(current_s), current_start, mt_vals[i-1]))
            current_s = stabs[i]; current_start = mt_vals[i]
    blocks.append((int(current_s), current_start, mt_vals[-1]))
    for i in range(1, len(blocks)-1):
        if blocks[i-1][0] == 2 and blocks[i][0] == 3 and blocks[i+1][0] == 2:
            lower_starts.append(blocks[i-1][1]); lower_ends.append(blocks[i-1][2])
            upper_starts.append(blocks[i+1][1]); upper_ends.append(blocks[i+1][2])
            mh_used2.append(mh_target)
            break
ax.fill_between(mh_used2, lower_starts, lower_ends, color='green', alpha=0.3, label='Lower 2 band')
ax.fill_between(mh_used2, upper_starts, upper_ends, color='green', alpha=0.3)
ax.plot(mh_used2, lower_starts, 'g--', lw=0.5)
ax.plot(mh_used2, lower_ends, 'g--', lw=0.5)
ax.plot(mh_used2, upper_starts, 'g-', lw=1)
ax.plot(mh_used2, upper_ends, 'g-', lw=1)
ax.set_xlabel(r'$M_h$ [GeV]')
ax.set_ylabel(r'$M_t$ [GeV]')
ax.set_title('(e) Metastable Band Boundaries')
ax.legend(fontsize=8)

# Panel 6: S_exact vs S_approx ratio through sandwich at Mh=5
ax = axes[1, 2]
mh_target, tol = 5.0, 0.02
chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt').copy()
for _, row in chunk.iterrows():
    if row['Stability'] == 2:
        ax.axvspan(row['Mt']-0.125, row['Mt']+0.125, alpha=0.1, color='yellow')
    elif row['Stability'] == 3:
        ax.axvspan(row['Mt']-0.125, row['Mt']+0.125, alpha=0.1, color='red')
mask = chunk['S_exact'] > 0 & (chunk['S_approx'] > 0)
ax.scatter(chunk['Mt'], chunk['S_exact']/chunk['S_approx'], c='purple', s=8, alpha=0.7)
ax.axhline(1, color='black', ls='-', lw=1, alpha=0.5)
ax.axhline(0, color='gray', ls='--', alpha=0.3)
ax.set_xlabel(r'$M_t$ [GeV]')
ax.set_ylabel(r'$S_{\rm exact} / S_{\rm approx}$')
ax.set_title(rf'(f) Action Ratio at $M_h = {mh_target}$ GeV')
ax.set_ylim(-2, 3)

plt.tight_layout()
plt.savefig('figures/sandwich_panel.png', dpi=300, bbox_inches='tight')
plt.close()
print('Panel figure saved: figures/sandwich_panel.png')

# --------------- ZOOM PLOT: Negative S region at Mh=100 ---------------
fig, ax = plt.subplots(figsize=(10, 6))
mh_target, tol = 100.0, 0.02
chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt').copy()
for _, row in chunk.iterrows():
    if row['Stability'] == 2:
        ax.axvspan(row['Mt']-0.125/2, row['Mt']+0.125/2, alpha=0.15, color='yellow')
    elif row['Stability'] == 3:
        ax.axvspan(row['Mt']-0.125/2, row['Mt']+0.125/2, alpha=0.15, color='red')
ax.scatter(chunk['Mt'], chunk['S_exact'], c='black', s=12, zorder=5)
# Connect points with line to show discontinuity
ax.plot(chunk['Mt'], chunk['S_exact'], 'gray', lw=0.5, alpha=0.3)
ax.axhline(0, color='blue', ls='--', lw=1, alpha=0.7, label=r'$S = 0$')
ax.set_xlabel(r'$M_t$ [GeV]')
ax.set_ylabel(r'$S_{\rm exact}$')
ax.set_title(rf'Action Discontinuity at $M_h = {mh_target}$ GeV (Fubini-Lipatov Breakdown)')
# annotate negative region
ax.annotate(f'S = {chunk.loc[chunk["S_exact"].idxmin(), "S_exact"]:.0f}',
           xy=(chunk.loc[chunk['S_exact'].idxmin(), 'Mt'], chunk.loc[chunk['S_exact'].idxmin(), 'S_exact']),
           xytext=(5, 20), textcoords='offset points', fontsize=9,
           arrowprops=dict(arrowstyle='->'))
ax.set_xlim(158, 161)
ax.set_ylim(-4000000, 50000)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/sandwich_zoom_Mh100.png', dpi=300, bbox_inches='tight')
plt.close()
print('Zoom figure saved: figures/sandwich_zoom_Mh100.png')

# --------------- SMOOTH PROFILE: Negative S across Mh ---------------
fig, ax = plt.subplots(figsize=(10, 6))
neg_fraction = []
mh_vals = np.arange(0, 130, 0.5)
for mh_target in mh_vals:
    tol = 0.02 if mh_target < 100 else 0.005
    chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)]
    n_neg = (chunk['S_exact'] < 0).sum()
    n_tot = len(chunk)
    neg_fraction.append(n_neg / n_tot * 100 if n_tot > 0 else 0)
ax.bar(mh_vals, neg_fraction, width=0.5, color='red', alpha=0.7)
ax.set_xlabel(r'$M_h$ [GeV]')
ax.set_ylabel(r'% of points with $S_{\rm exact} < 0$')
ax.set_title('Fraction of Negative-Action Points vs Higgs Mass')
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/sandwich_neg_fraction.png', dpi=300, bbox_inches='tight')
plt.close()
print('Negative fraction figure saved: figures/sandwich_neg_fraction.png')

# --------------- SUMMARY STATISTICS ---------------
total = len(df)
computed = df[df['S_exact'] != -1.0]
n_neg = (computed['S_exact'] < 0).sum()
n_pos = (computed['S_exact'] > 0).sum()
n_in_sw = 0
n_in_sw_neg = 0
for mh_target in np.arange(0, 135, 0.25):
    tol = 0.02 if mh_target < 100 else 0.01
    chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)]
    if len(chunk) < 10: continue
    stabs = chunk['Stability'].values.astype(int)
    mt_vals = chunk['Mt'].values
    blocks = []
    current_s = stabs[0]; current_start = mt_vals[0]
    for i in range(1, len(stabs)):
        if stabs[i] != current_s:
            blocks.append((int(current_s), current_start, mt_vals[i-1]))
            current_s = stabs[i]; current_start = mt_vals[i]
    blocks.append((int(current_s), current_start, mt_vals[-1]))
    for i in range(1, len(blocks)-1):
        if blocks[i-1][0] == 2 and blocks[i][0] == 3 and blocks[i+1][0] == 2:
            mask = (chunk['Mt'] >= blocks[i][1]) & (chunk['Mt'] <= blocks[i][2])
            in_sw = chunk[mask]
            n_in_sw += len(in_sw)
            n_in_sw_neg += (in_sw['S_exact'] < 0).sum()
            break

print(f"\n=== SUMMARY ===")
print(f"Total points in CSV: {total}")
print(f"Total computed (S_exact != -1): {len(computed)}")
print(f"  Positive S_exact: {n_pos}")
print(f"  Negative S_exact: {n_neg}")
print(f"  Fraction negative: {100*n_neg/len(computed):.2f}%")
print(f"Points in sandwiched-3 region: {n_in_sw}")
print(f"  Negative S in sandwiched-3: {n_in_sw_neg}")
print(f"  Fraction negative in sandwich: {100*n_in_sw_neg/n_in_sw:.1f}%" if n_in_sw > 0 else "  N/A")
print(f"Sandwich present at M_h: 0 to ~135 GeV")
print(f"Max negative S in sandwich: S = {computed['S_exact'].min():.0f}")
