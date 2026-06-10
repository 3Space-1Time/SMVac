import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
})

DARK = '#1a1a2e'
WHITE = '#E0E0E0'
GOLD = '#FFD54F'

def generate_plot(csv_file, out_file, title, is_zoomed=False):
    print(f"Loading boundary data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File {csv_file} not found. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=DARK)
    ax.set_facecolor(DARK)

    Mt = df['Mt']
    Mh_abs = df['Mh_abs']
    Mh_c6_0 = df['Mh_c6_0']
    Mh_c6_01 = df['Mh_c6_0_1']
    Mh_c6_10 = df['Mh_c6_1_0']

    # 1. Fill Regions
    # Unstable region for c6=1.0 (the most stabilized case)
    ax.fill_betweenx(Mt, 0, Mh_c6_10, color='#F44336', alpha=0.3, label=r'Unstable ($c_6=1.0$)')
    
    # The region that WAS unstable in SM but is now metastable due to c6=1.0
    ax.fill_betweenx(Mt, Mh_c6_10, Mh_c6_0, color='#9C27B0', alpha=0.4, label='Stabilized by $c_6=1.0$')
    
    # Normal SM Metastable region
    ax.fill_betweenx(Mt, Mh_c6_0, Mh_abs, color='#2196F3', alpha=0.4, label='Metastable (SM)')
    
    # Stable region
    ax.fill_betweenx(Mt, Mh_abs, 300, color='#4CAF50', alpha=0.2, label='Stable')

    # 2. Boundary Lines
    ax.plot(Mh_abs, Mt, color=WHITE, ls='-', lw=3.0, zorder=10, label='Absolute Stability Boundary')
    
    ax.plot(Mh_c6_0, Mt, color='#64B5F6', ls='-', lw=2.5, zorder=5, label=r'Instability ($c_6=0$)')
    ax.plot(Mh_c6_01, Mt, color=GOLD, ls='--', lw=2.0, zorder=7, label=r'Instability ($c_6=0.1$)')
    ax.plot(Mh_c6_10, Mt, color='#EF5350', ls='-', lw=2.5, zorder=6, label=r'Instability ($c_6=1.0$)')

    # Formatting
    if is_zoomed:
        ax.set_xlim(110.0, 135.0)
        ax.set_ylim(165.0, 180.0)
    else:
        ax.set_xlim(0.0, 300.0)
        ax.set_ylim(0.0, 300.0)

    ax.set_ylabel(r'Top Quark Mass $M_t$ [GeV]', color=WHITE)
    ax.set_xlabel(r'Higgs Mass $M_h$ [GeV]', color=WHITE)
    ax.set_title(title, color=WHITE, fontweight='bold', pad=15)

    ax.tick_params(colors=WHITE)
    for sp in ax.spines.values():
        sp.set_color('#555555')
    ax.grid(True, color='#2a2a4a', ls='--', lw=0.5)

    # SM Point
    if is_zoomed or True:
        ax.annotate('Standard Model', xy=(125.25, 172.5), xytext=(128.0, 169.0) if is_zoomed else (145.0, 165.0),
                    color=GOLD, fontsize=12, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=GOLD, lw=2), ha='center')
        ax.plot([125.25], [172.5], marker='*', color=GOLD, markersize=14, zorder=20)

    ax.legend(facecolor='#1a1a2e', edgecolor='#555555', labelcolor=WHITE, loc='upper left', framealpha=0.9, fontsize=10)

    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor=DARK)
    print(f"Saved plot to {out_file}")
    plt.close()

if __name__ == "__main__":
    generate_plot(
        'results/c6_boundaries_broad.csv',
        'results/c6_phase_diagram_broad.png',
        r'Full Phase Diagram: Expanding Metastability due to Planck-Scale $c_6$',
        is_zoomed=False
    )
    
    generate_plot(
        'results/c6_boundaries_zoomed.csv',
        'results/c6_phase_diagram_zoomed.png',
        r'Zoomed Phase Diagram: Expanding Metastability due to Planck-Scale $c_6$',
        is_zoomed=True
    )
