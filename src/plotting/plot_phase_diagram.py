import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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

print("Reading data...")
try:
    df = pd.read_csv('../../results/data.csv')
except FileNotFoundError:
    print("Error: 'data.csv' not found.")
    exit()

# Filter out massive outliers
df_filtered = df[(df['Mh'] <= 1000) & (df['Mt'] <= 1000)].copy()

# Extract arrays for plotting
X = df_filtered['Mh'].values
Y = df_filtered['Mt'].values
Z = df_filtered['Stability'].values

# Define exact colors from the reference paper
# 1=Stable (Green), 2=Metastable (Yellow), 3=Unstable (Red), 4=Non-perturbative (Grey)
cmap = mcolors.ListedColormap(['#78f078', '#ffff66', '#ff7878', '#505050'])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5] 
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(12, 9))

# 1. Plot your C++ data as a dense scatter plot
contour = plt.scatter(X, Y, c=Z, cmap=cmap, norm=norm, s=4.0, marker='s', edgecolors='none')



# 3. Plot the Standard Model Vacuum (Experimental Point)
sm_mh = 125.10
sm_mt = 173.1
plt.plot(sm_mh, sm_mt, marker='*', markersize=15, color='navy', zorder=5, label='SM Central Value')

sigma_mh = 0.14
sigma_mt = 0.6
ax1 = plt.gca()
for n, alpha in zip([3, 2, 1], [0.15, 0.35, 0.55]):
    ellipse = mpatches.Ellipse((sm_mh, sm_mt), width=2*n*sigma_mh, height=2*n*sigma_mt, 
                               facecolor=mcolors.to_rgba('dodgerblue', alpha), 
                               edgecolor='navy', linewidth=1.5, zorder=4, 
                               label=f'{n}$\\sigma$ CL')
    ax1.add_patch(ellipse)

# Labels and Styling
plt.xlabel('Higgs pole mass $M_h$ in GeV', fontsize=14)
plt.ylabel('Top pole mass $M_t$ in GeV', fontsize=14)
plt.xlim(0, 250)
plt.ylim(0, 250)
plt.title('Stability Phase Diagram: C++ Calculation', fontsize=16)

# Custom Legend for Phase Regions
patch_stable = mpatches.Patch(color='#78f078', label='Stability')
patch_meta = mpatches.Patch(color='#ffff66', label='Meta-stability')
patch_unstable = mpatches.Patch(color='#ff7878', label='Instability')
patch_nonpert = mpatches.Patch(color='#505050', label='Non-perturbative')

handles, labels = plt.gca().get_legend_handles_labels()
handles.extend([patch_stable, patch_meta, patch_unstable, patch_nonpert])

plt.legend(handles=handles, loc='lower right', framealpha=0.9, edgecolor='black')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

plt.savefig('../../results/stability_plot.png', dpi=300)
print("Saved smoothed plot to ../../results/stability_plot.png")

# ---------------------------------------------------------
# FIGURE 2: Close-up Phase Diagram with Error Ellipses
# ---------------------------------------------------------
from matplotlib.patches import Ellipse

plt.figure(figsize=(8, 8))

# Plot scatter again for the close-up
plt.scatter(X, Y, c=Z, cmap=cmap, norm=norm, s=80.0, marker='s', edgecolors='none')

# SM Vacuum Central Value
plt.plot(sm_mh, sm_mt, marker='*', markersize=15, color='navy', zorder=5, label='SM Central Value')

# Current PDG approximate uncertainties
sigma_mh = 0.14
sigma_mt = 0.6

ax = plt.gca()
# Plot 3, 2, 1 sigma ellipses
# width and height are full axes lengths, so 2 * n * sigma
for n, alpha in zip([3, 2, 1], [0.15, 0.35, 0.55]):
    ellipse = Ellipse((sm_mh, sm_mt), width=2*n*sigma_mh, height=2*n*sigma_mt, 
                      facecolor=mcolors.to_rgba('dodgerblue', alpha), 
                      edgecolor='navy', linewidth=1.5, zorder=4, 
                      label=f'{n}$\\sigma$ CL')
    ax.add_patch(ellipse)

# Limits for close-up (Expanded by 3x)
plt.xlim(121.5, 129.0)
plt.ylim(165.0, 180.0)

plt.xlabel('Higgs pole mass $M_h$ in GeV')
plt.ylabel('Top pole mass $M_t$ in GeV')
plt.title('Close-up: SM Vacuum $\\sigma$ Contours')

# Update handles for close-up
handles2, labels2 = ax.get_legend_handles_labels()
# Add the phase colors (usually only Meta-stability and Stability are visible here)
handles2.extend([patch_stable, patch_meta])

plt.legend(handles=handles2, loc='upper left', framealpha=0.9, edgecolor='black')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('../../results/stability_closeup_plot.png', dpi=300)
print("Saved close-up plot to ../../results/stability_closeup_plot.png")