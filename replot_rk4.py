"""
replot_rk4.py — Replot Figure 1 (RK4 convergence) with correct axis orientation.
Step size increases LEFT → RIGHT. Error increases LEFT → RIGHT. Orthodox.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

DARK   = '#1a1a2e'
BLUE   = '#4FC3F7'
RED    = '#EF5350'
GOLD   = '#FFD54F'
GREEN  = '#81C784'
WHITE  = '#E0E0E0'
ORANGE = '#FFA726'
TEAL   = '#4DB6AC'

# ─── Model ODE: dy/dt = -k*y, y(0)=1, exact = exp(-k*t) ────────────────────
# Represents the smooth exponential decay of the Higgs quartic coupling lambda(t)
k     = 0.05
t0    = 0.0
t_end = 100.0
y_exact_end = math.exp(-k * t_end)

# Step sizes from FINE to COARSE (will plot left=fine, right=coarse)
dts = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

rk4_errs   = []
euler_errs  = []
rk2_errs   = []   # midpoint method (2nd order), for reference

for dt in dts:
    # --- RK4 ---
    y, t = 1.0, t0
    while t < t_end - 1e-12:
        h  = min(dt, t_end - t)
        k1 = -k * y
        k2 = -k * (y + 0.5*h*k1)
        k3 = -k * (y + 0.5*h*k2)
        k4 = -k * (y + h*k3)
        y  = y + h/6.0*(k1 + 2*k2 + 2*k3 + k4)
        t += h
    rk4_errs.append(abs(y - y_exact_end))

    # --- Euler (1st order) ---
    y, t = 1.0, t0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        y = y + h * (-k * y)
        t += h
    euler_errs.append(abs(y - y_exact_end))

    # --- RK2 midpoint (2nd order) ---
    y, t = 1.0, t0
    while t < t_end - 1e-12:
        h  = min(dt, t_end - t)
        k1 = -k * y
        k2 = -k * (y + 0.5*h*k1)
        y  = y + h * k2
        t += h
    rk2_errs.append(abs(y - y_exact_end))

# ─── Reference slope lines anchored at the FINE end ─────────────────────────
dt_arr = np.array(dts, dtype=float)
# Anchor at dt=0.05 (middle of range, in asymptotic regime for all methods)
i_anchor = dts.index(0.05)

slope4_anchor = rk4_errs[i_anchor]
slope2_anchor = rk2_errs[i_anchor]
slope1_anchor = euler_errs[i_anchor]

slope4 = slope4_anchor * (dt_arr / dts[i_anchor])**4
slope2 = slope2_anchor * (dt_arr / dts[i_anchor])**2
slope1 = slope1_anchor * (dt_arr / dts[i_anchor])**1

# ─── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=DARK)
ax.set_facecolor(DARK)

ax.loglog(dts, rk4_errs,   'o-', color=BLUE,   lw=2.2, ms=7,
          label='RK4 — 4th order', zorder=5)
ax.loglog(dts, rk2_errs,   's-', color=TEAL,   lw=2.0, ms=6,
          label='RK2 midpoint — 2nd order', zorder=4)
ax.loglog(dts, euler_errs, '^-', color=RED,    lw=2.0, ms=6,
          label='Euler — 1st order', zorder=3)

# Reference slopes
ax.loglog(dt_arr, slope4, '--', color=GOLD,   lw=1.4, alpha=0.85, label=r'$\propto h^{4}$ (expected RK4)')
ax.loglog(dt_arr, slope2, '--', color=GREEN,  lw=1.4, alpha=0.85, label=r'$\propto h^{2}$')
ax.loglog(dt_arr, slope1, '--', color=ORANGE, lw=1.4, alpha=0.85, label=r'$\propto h^{1}$')

# Annotate our production step sizes
for h_mark, label, col in [(0.01, r'$\Delta t = 0.01$'+'\n(high-prec run)', BLUE),
                             (0.1,  r'$\Delta t = 0.1$'+'\n(previous)', RED)]:
    err_idx = dts.index(h_mark)
    ax.annotate(label,
                xy=(h_mark, rk4_errs[err_idx]),
                xytext=(h_mark * 0.22, rk4_errs[err_idx] * 25),
                color=col, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

ax.set_xlabel(r'Integration step size  $h = \Delta t$  [increasing $\rightarrow$ coarser]',
              color=WHITE)
ax.set_ylabel(r'Global truncation error  $|y(T) - y_{exact}|$', color=WHITE)
ax.set_title('RK4 Convergence Study on Model $\\beta$-Function ODE\n'
             r'$dy/dt = -0.05\,y$,  $y(0)=1$,  $T=100$,  $y_{exact}=e^{-5}$',
             color=WHITE, fontweight='bold')

ax.tick_params(colors=WHITE, which='both')
for sp in ax.spines.values():
    sp.set_color('#444')

# Grid
ax.grid(True, which='major', color='#333', ls='-',  lw=0.6)
ax.grid(True, which='minor', color='#222', ls='--', lw=0.4)

# Direction annotation
ax.annotate('Coarser grid\n(larger error)',
            xy=(0.92, 0.88), xycoords='axes fraction',
            color='#aaa', fontsize=9, ha='right')
ax.annotate('Finer grid\n(smaller error)',
            xy=(0.10, 0.12), xycoords='axes fraction',
            color='#aaa', fontsize=9, ha='left')

ax.legend(facecolor='#1e1e38', edgecolor='#555', labelcolor=WHITE,
          loc='upper left', framealpha=0.85)

plt.tight_layout()
fig.savefig('results/error_rk4_convergence.png', dpi=200,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("Saved results/error_rk4_convergence.png")
