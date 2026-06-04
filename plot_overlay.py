import pandas as pd
import matplotlib.pyplot as plt

def create_overlay_plot():
    df_ana = pd.read_csv('results/analytical_data.csv')
    df_num = pd.read_csv('results/numerical_data.csv')

    plt.figure(figsize=(10, 8))

    # Analytical regions
    ana_metastable = df_ana[df_ana['Stability'] == 2]
    ana_unstable = df_ana[df_ana['Stability'] == 3]

    # Numerical regions
    num_metastable = df_num[df_num['Stability'] == 2]
    num_unstable = df_num[df_num['Stability'] == 3]

    # Overlay colors: We will just plot lines to show boundaries, or plot them as scattered domains.
    # To see the difference, it's easiest to plot one as solid points and the other as hollow points, 
    # but with 400k points that's too heavy. Instead, let's plot the boundaries.
    
    # Let's plot the numerical as the base color
    plt.scatter(df_num[df_num['Stability'] == 1]['Mh_calc'], df_num[df_num['Stability'] == 1]['Mt'], c='green', alpha=0.3, s=5, label='Stable (Both)')
    plt.scatter(num_metastable['Mh_calc'], num_metastable['Mt'], c='yellow', alpha=0.3, s=5, label='Metastable (Numerical)')
    plt.scatter(num_unstable['Mh_calc'], num_unstable['Mt'], c='red', alpha=0.3, s=5, label='Unstable (Numerical)')
    
    # We can plot the Analytical unstable region points that are metastable in numerical.
    # We will identify the points where analytical and numerical DISAGREE.
    merged = pd.merge(df_ana, df_num, on=['Mt', 'Mh_calc'], suffixes=('_ana', '_num'))
    
    diff = merged[merged['Stability_ana'] != merged['Stability_num']]
    
    if not diff.empty:
        plt.scatter(diff['Mh_calc'], diff['Mt'], c='blue', alpha=1.0, s=5, label='Difference (Boundary Shift)')
    
    plt.scatter(125.10, 173.1, color='blue', marker='*', s=200, edgecolor='black', label='Experimental (125.1, 173.1)')

    plt.xlabel('Higgs Mass $M_h$ (GeV)')
    plt.ylabel('Top Mass $M_t$ (GeV)')
    plt.title('Vacuum Stability Phase Diagram Overlay (Analytical vs Numerical)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('results/overlay_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Close-up plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df_num[df_num['Stability'] == 1]['Mh_calc'], df_num[df_num['Stability'] == 1]['Mt'], c='green', alpha=0.3, s=5)
    plt.scatter(num_metastable['Mh_calc'], num_metastable['Mt'], c='yellow', alpha=0.3, s=5)
    plt.scatter(num_unstable['Mh_calc'], num_unstable['Mt'], c='red', alpha=0.3, s=5)
    
    if not diff.empty:
        plt.scatter(diff['Mh_calc'], diff['Mt'], c='blue', alpha=1.0, s=15, label='Difference (Boundary Shift)')
        
    plt.scatter(125.10, 173.1, color='blue', marker='*', s=300, edgecolor='black', label='Experimental (125.1, 173.1)')
    plt.xlim(110, 140)
    plt.ylim(160, 185)
    
    plt.xlabel('Higgs Mass $M_h$ (GeV)')
    plt.ylabel('Top Mass $M_t$ (GeV)')
    plt.title('Vacuum Stability Phase Diagram Overlay (Close-up)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('results/overlay_closeup_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_overlay_plot()
    print("Overlay plots generated.")
