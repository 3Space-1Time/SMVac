import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 7))

df_u = pd.read_csv("data/bisection_undershoot.csv")
df_c = pd.read_csv("data/bisection_converged.csv")
df_o = pd.read_csv("data/bisection_overshoot.csv")

plt.plot(df_u['r'], df_u['phi'], label='Undershoot (\u03d5_0 \u2248 2.326e4)', color='blue', linestyle='--')
plt.plot(df_o['r'], df_o['phi'], label='Overshoot (\u03d5_0 \u2248 2.743e4)', color='red', linestyle='--')
plt.plot(df_c['r'], df_c['phi'], label='Converged Bounce (\u03d5_0 \u2248 2.373e4)', color='green', linewidth=2)

plt.xscale('log')
plt.yscale('symlog', linthresh=1e-2)
plt.xlabel('Radius r (1/GeV)')
plt.ylabel('Field \u03d5(r) (GeV)')
plt.title('Converged Canonical Bounce for Mh=5 GeV, Mt=105 GeV')
plt.axhline(0, color='black', linewidth=1, ls='-')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("figures/bisection_plot.png", dpi=300)
print("Saved plot to figures/bisection_plot.png")
