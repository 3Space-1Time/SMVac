import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

# Academic styling
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.formatter.use_mathtext': True
})

print("Reading dense analytical data...")
df = pd.read_csv('results/analytical_data_dense.csv')

# Filter out massive outliers
df_filtered = df[(df['Mh_calc'] <= 1000) & (df['Mt'] <= 1000)].copy()

X = df_filtered['Mh_calc'].values
Y = df_filtered['Mt'].values
Z = df_filtered['Stability'].values

# Define exact colors from the reference paper
# 1=Stable (Green), 2=Metastable (Yellow), 3=Unstable (Red), 4=Non-perturbative (Grey)
cmap = mcolors.ListedColormap(['#78f078', '#ffff66', '#ff7878', '#505050'])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5] 
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(10, 8))

# Dense scatter plot
plt.scatter(X, Y, c=Z, cmap=cmap, norm=norm, s=150.0, marker='s', edgecolors='none')

# SM Vacuum Central Value
sm_mh = 125.10
sm_mt = 173.1
plt.plot(sm_mh, sm_mt, marker='*', markersize=15, color='navy', zorder=5, label='SM Central Value')

sigma_mh = 0.14
sigma_mt = 0.6
ax1 = plt.gca()
for n, alpha in zip([3, 2, 1], [0.15, 0.35, 0.55]):
    ellipse = Ellipse((sm_mh, sm_mt), width=2*n*sigma_mh, height=2*n*sigma_mt, 
                               facecolor=mcolors.to_rgba('dodgerblue', alpha), 
                               edgecolor='navy', linewidth=1.5, zorder=4, 
                               label=f'{n}$\\sigma$ CL')
    ax1.add_patch(ellipse)

# Labels and Styling
plt.xlabel('Higgs pole mass $M_h$ in GeV', fontsize=16)
plt.ylabel('Top pole mass $M_t$ in GeV', fontsize=16)

# Set limits around the SM point to mimic Buttazzo
plt.xlim(118, 130)
plt.ylim(168, 178)
plt.title('Vacuum Stability in the SM (Analytical)', fontsize=18)

# Custom Legend for Phase Regions
patch_stable = mpatches.Patch(color='#78f078', label='Absolute Stability')
patch_meta = mpatches.Patch(color='#ffff66', label='Metastability')
patch_unstable = mpatches.Patch(color='#ff7878', label='Instability')

handles, labels = plt.gca().get_legend_handles_labels()
# Add the phase colors
handles.extend([patch_stable, patch_meta, patch_unstable])

plt.legend(handles=handles, loc='upper left', framealpha=0.9, edgecolor='black')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

output_path = 'results/buttazzo_plot.png'
plt.savefig(output_path, dpi=300)
print(f"Saved smoothed dense plot to {output_path}")
