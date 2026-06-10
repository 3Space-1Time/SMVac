import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Reading precision data...")
try:
    df = pd.read_csv('../../results/precision_data.csv')
except FileNotFoundError:
    print("Error: 'precision_data.csv' not found.")
    exit()

# Filter out points that are stable or failed (represented by -1)
df_valid = df[(df['S_exact'] > 0) & (df['S_approx'] > 0)].copy()

if df_valid.empty:
    print("No valid unstable points found in the scanned region.")
    exit()

# Calculate Percentage Error
# Error = |S_exact - S_approx| / S_exact * 100
df_valid['Error_Pct'] = np.abs(df_valid['S_exact'] - df_valid['S_approx']) / df_valid['S_exact'] * 100.0

X = df_valid['Mh'].values
Y = df_valid['Mt'].values
Z_err = df_valid['Error_Pct'].values

plt.figure(figsize=(10, 8))
# Use a scatter plot or tricontourf
contour = plt.tricontourf(X, Y, Z_err, levels=50, cmap='inferno')
cbar = plt.colorbar(contour)
cbar.set_label('Absolute Percentage Error (%)', fontsize=12)

# Mark the SM Vacuum
sm_mh = 125.10
sm_mt = 173.1
plt.plot(sm_mh, sm_mt, marker='*', markersize=15, color='white', markeredgecolor='black', label='SM Vacuum (125.1, 173.1)')

plt.xlabel('Higgs pole mass $M_h$ [GeV]', fontsize=14)
plt.ylabel('Top pole mass $M_t$ [GeV]', fontsize=14)
plt.title('Accuracy of Analytical Bounce Approx vs Exact SimpleBounce', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('../../results/precision_error_heatmap.png', dpi=300)
print("Saved heatmap to ../../results/precision_error_heatmap.png")

# Also plot the actual S_exact values to show the S=400 contour
plt.figure(figsize=(10, 8))
Z_exact = df_valid['S_exact'].values
contour2 = plt.tricontourf(X, Y, Z_exact, levels=50, cmap='viridis')
cbar2 = plt.colorbar(contour2)
cbar2.set_label('Exact Bounce Action $S_4$', fontsize=12)

# Overlay the dynamic Metastability boundary (where Stability transitions from 2 to 3)
Z_stab = df_valid['Stability'].values
plt.tricontour(X, Y, Z_stab, levels=[2.5], colors='red', linewidths=2, linestyles='--')
plt.plot([],[], color='red', linestyle='--', linewidth=2, label='Exact Metastability Boundary')

plt.plot(sm_mh, sm_mt, marker='*', markersize=15, color='white', markeredgecolor='black', label='SM Vacuum')

plt.xlabel('Higgs pole mass $M_h$ [GeV]', fontsize=14)
plt.ylabel('Top pole mass $M_t$ [GeV]', fontsize=14)
plt.title('Exact Bounce Action $S_4$ around SM Vacuum', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('../../results/precision_exact_action.png', dpi=300)
print("Saved exact action plot to ../../results/precision_exact_action.png")
