import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('../../results/data.csv')
except FileNotFoundError:
    print("Run ./run.sh first!")
    exit()

# Filter physical range
df = df[(df['Calculated_Mh'] <= 200) & (df['Mt'] <= 200)]

colors = {1: 'green', 2: '#F4D03F', 3: 'red', 4: 'black'}
labels = {1: 'Stable', 2: 'Metastable', 3: 'Unstable', 4: 'Non-perturbative'}

plt.figure(figsize=(10, 8))

for code in sorted(df['Stability'].unique()):
    subset = df[df['Stability'] == code]
    if len(subset) == 0: continue
    plt.scatter(subset['Calculated_Mh'], subset['Mt'], c=colors.get(code,'gray'), 
                label=labels.get(code), s=20, edgecolors='none', alpha=0.7)

# Experimental Star
plt.scatter(125.1, 173.1, marker='*', s=350, c='white', edgecolors='black', 
            linewidth=1.5, label='Experimental', zorder=10)

plt.title('Standard Model Vacuum Stability (Gradient Flow)', fontsize=14)
plt.xlabel('Higgs Mass $M_h$ [GeV]', fontsize=12)
plt.ylabel('Top Mass $M_t$ [GeV]', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(1, 300)
plt.ylim(1, 300)

plt.savefig('../../results/stability_plot.png', dpi=300)
print("Saved ../../results/stability_plot.png")