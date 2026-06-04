import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df_disc = pd.read_csv('../../results/discretization_error.csv')
df_vol = pd.read_csv('../../results/finite_volume_error.csv')

# 1. Discretization Error Processing
# Assume the highest N (last row) is the exact Ground Truth
S_true_disc = df_disc['S_exact'].iloc[-1]
df_disc['Error'] = np.abs(df_disc['S_exact'] - S_true_disc)

# 2. Finite Volume Error Processing
S_true_vol = df_vol['S_exact'].iloc[-1]
df_vol['Error'] = np.abs(df_vol['S_exact'] - S_true_vol)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Left Panel: Discretization Error
axs[0].plot(df_disc['N'][:-1], df_disc['Error'][:-1], marker='o', linestyle='-', color='blue', label='Measured Error')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel('Number of Grid Points $N$', fontsize=14)
axs[0].set_ylabel('Absolute Error $|S_N - S_{max}|$', fontsize=14)
axs[0].set_title('Discretization Error Scaling', fontsize=16)

# Plot perfect O(1/N^2) reference line
N_ref = np.array(df_disc['N'][:-1])
# Shift reference line slightly so it's visible next to the data
offset = df_disc['Error'].iloc[0] * (N_ref[0]**2)
axs[0].plot(N_ref, offset / (N_ref**2), linestyle='--', color='gray', label='$\mathcal{O}(1/N^2)$ Theoretical Scaling')
axs[0].legend(fontsize=12)
axs[0].grid(True, which="both", ls="--", alpha=0.5)

# Right Panel: Finite Volume Error
axs[1].plot(df_vol['Rmax'][:-1], df_vol['Error'][:-1], marker='s', linestyle='-', color='red', label='Measured Error')
axs[1].set_yscale('log')
axs[1].set_xlabel('Boundary Cutoff Radius $r_{max}$', fontsize=14)
axs[1].set_ylabel('Absolute Error $|S_{R_{max}} - S_{R=2.0}|$', fontsize=14)
axs[1].set_title('Finite Volume Error Scaling', fontsize=16)
axs[1].legend(fontsize=12)
axs[1].grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig('../../results/error_scaling_plots.png', dpi=300)
print("Saved plots to ../../results/error_scaling_plots.png")
