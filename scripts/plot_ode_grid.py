import pandas as pd
import matplotlib.pyplot as plt

def create_plot(df, filename, title, xlim=None, ylim=None, s=5):
    stable = df[df['Stability'] == 1]
    metastable = df[df['Stability'] == 2]
    unstable = df[df['Stability'] == 3]
    nonpert = df[df['Stability'] == 4]

    plt.figure(figsize=(10, 8))
    
    plt.scatter(stable['Mh_calc'], stable['Mt'], c='green', alpha=0.5, s=s, label='Stable')
    plt.scatter(metastable['Mh_calc'], metastable['Mt'], c='yellow', alpha=0.5, s=s, label='Metastable')
    plt.scatter(unstable['Mh_calc'], unstable['Mt'], c='red', alpha=0.5, s=s, label='Unstable')
    
    if not nonpert.empty:
        plt.scatter(nonpert['Mh_calc'], nonpert['Mt'], c='black', alpha=0.5, s=s, label='Non-perturbative')

    plt.scatter(125.10, 173.1, color='blue', marker='*', s=200, edgecolor='black', label='Experimental (125.1, 173.1)', zorder=5)

    plt.xlabel('Higgs Mass $M_h$ (GeV)')
    plt.ylabel('Top Mass $M_t$ (GeV)')
    plt.title(title)
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
        
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

print("Plotting full view using ODE grid data...")
try:
    df_full = pd.read_csv('data/ode_grid_data.csv')
    create_plot(df_full, 'figures/ode_stability_plot.png', 'Standard Model Vacuum Stability (Exact ODE, Full View)', s=5)
    
    df_partial = df_full[(df_full['Mh_calc'] <= 140) & (df_full['Mt'] <= 185)]
    create_plot(df_partial, 'figures/ode_stability_plot_partial.png', 'Standard Model Vacuum Stability (Exact ODE, Partial View)', xlim=(0, 140), ylim=(100, 185), s=5)
except FileNotFoundError:
    print("data/ode_grid_data.csv not found yet.")
