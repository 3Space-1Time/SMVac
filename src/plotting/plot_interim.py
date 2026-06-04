import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load interim data
data = pd.read_csv('c:/Users/LENOVO/code/Threshold/results/data_interim.csv')

# C++ code classification:
# 1 = Stable
# 2 = Metastable
# 3 = Unstable
# 4 = Non-perturbative

stable_points = data[data['Stability'] == 1]
metastable_points = data[data['Stability'] == 2]
unstable_points = data[data['Stability'] == 3]
nonperturb_points = data[data['Stability'] == 4]

print(f"Non-perturb points: {len(nonperturb_points)}")
if not nonperturb_points.empty:
    print(f"Non-perturb Mh min: {nonperturb_points['Mh'].min()}, max: {nonperturb_points['Mh'].max()}")

plt.figure(figsize=(10, 8))
if not stable_points.empty:
    plt.scatter(stable_points['Mh'], stable_points['Mt'], c='blue', s=10, label='Stable', alpha=0.5)
if not metastable_points.empty:
    plt.scatter(metastable_points['Mh'], metastable_points['Mt'], c='green', s=10, label='Metastable', alpha=0.5)
if not unstable_points.empty:
    plt.scatter(unstable_points['Mh'], unstable_points['Mt'], c='red', s=10, label='Unstable', alpha=0.5)
if not nonperturb_points.empty:
    plt.scatter(nonperturb_points['Mh'], nonperturb_points['Mt'], c='black', s=20, label='Non-Perturbative', alpha=1.0, marker='x')

plt.xlabel('Higgs Mass (GeV)', fontsize=14)
plt.ylabel('Top Mass (GeV)', fontsize=14)
plt.title('SM Stability Phase Diagram (Interim - CORRECTED)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('C:/Users/LENOVO/.gemini/antigravity-ide/brain/4adf140e-99da-4f14-aba4-7368a9a04589/stability_interim.png', dpi=300, bbox_inches='tight')
print("Plot saved.")
