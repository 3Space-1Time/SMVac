"""Correct potential plot with proper minimization condition V'(v)=0."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13

v = 246.22
pi = np.pi
PI2 = pi**2
Mt = 100.0; Mh = 5.0

# NNLO matching
lam_tree = Mh**2 / (2.0*v**2)
yt = np.sqrt(2.0)*Mt/v - 0.0587
d_lam = -0.00313 - 0.00004*(Mt-173.34)
lam = lam_tree + d_lam
g1 = 0.46266 + 0.00006*(Mt-173.34)
g2 = 0.65355 + 0.00002*(Mt-173.34)

# CW 1-loop correction
def V1_loop(phi):
    Mt2 = 0.5 * yt**2 * phi**2
    Mw2 = 0.25 * g2**2 * phi**2
    Mz2 = 0.25 * (g1**2 + g2**2) * phi**2
    mu_ren2 = Mt**2
    V = 0.0
    if Mt2 > 0:
        V += -12.0 * Mt2**2 * (np.log(Mt2/mu_ren2) - 1.5)
    if Mw2 > 0:
        V += 6.0 * Mw2**2 * (np.log(Mw2/mu_ren2) - 5.0/6.0)
    if Mz2 > 0:
        V += 3.0 * Mz2**2 * (np.log(Mz2/mu_ren2) - 5.0/6.0)
    return V / (64.0 * PI2)

# Correct mu^2 from V'(v) = 0
h = 1e-4
V1p_v = (V1_loop(v+h) - V1_loop(v-h)) / (2*h)
V1pp_v = (V1_loop(v+h) - 2*V1_loop(v) + V1_loop(v-h)) / (h**2)
mu2 = lam * v**2 + V1p_v / v

Mh2_phys = -mu2 + 3*lam*v**2 + V1pp_v
print(f"lam(Mt) = {lam:.6e}")
print(f"mu^2 (correct) = {mu2:.4f} GeV^2")
print(f"Mh_phys = {np.sqrt(abs(Mh2_phys)):.2f} GeV  (note: differs from input due to scheme)")
print(f"V''(v) > 0 => local minimum: {Mh2_phys > 0}")

def V_total(phi):
    return -mu2/2.0 * phi**2 + lam/4.0 * phi**4 + V1_loop(phi)

# Verify
Vp = (V_total(v+h) - V_total(v-h)) / (2*h)
print(f"V'(v) = {Vp:.2e}  (should be 0)")

# ── Find the barrier ──
phi_scan = np.linspace(v, 5000, 200000)
V_scan = np.array([V_total(p) for p in phi_scan])
V_at_v = V_total(v)

# Find local max after vev
dV = np.diff(V_scan)
# Find where slope changes from positive to negative (barrier top)
slope_sign = np.sign(dV)
sign_changes = np.where(np.diff(slope_sign) < 0)[0]  # positive -> negative = peak

if len(sign_changes) > 0:
    idx_bar = sign_changes[0] + 1
    phi_bar = phi_scan[idx_bar]
    V_bar = V_scan[idx_bar]
    barrier_height = V_bar - V_at_v
    print(f"\nBarrier top at phi = {phi_bar:.1f} GeV")
    print(f"V(barrier) = {V_bar:.4e} GeV^4")
    print(f"V(v)       = {V_at_v:.4e} GeV^4")
    print(f"Barrier height = {barrier_height:.4e} GeV^4")
    print(f"Barrier height / v^4 = {barrier_height/v**4:.4e}")
    has_barrier = True
else:
    print("\nNo barrier found!")
    has_barrier = False
    phi_bar = v + 500

# Find where potential crosses zero (becomes more negative than V=0)
zero_crossings = np.where(np.diff(np.sign(V_scan)))[0]
if len(zero_crossings) > 0:
    phi_zero = phi_scan[zero_crossings[0]]
    print(f"V crosses zero at phi = {phi_zero:.1f} GeV")

# ── PLOTS ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    rf'Corrected EW Potential ($V^\prime(v)=0$ enforced):  '
    rf'$M_t = 100$ GeV, $M_h = 5$ GeV, $\lambda(M_t) = {lam:.1e}$',
    fontsize=14, fontweight='bold')

# Panel 1: Full view 0-5000 GeV (linear)
ax = axes[0, 0]
phi_wide = np.linspace(1, 5000, 10000)
V_wide = np.array([V_total(p) for p in phi_wide])
ax.plot(phi_wide, V_wide/v**4, 'r-', lw=2, label='Full 1-loop potential')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(v, color='green', ls=':', lw=1.5, alpha=0.7, label=f'v = {v:.0f} GeV')
if has_barrier:
    ax.axvline(phi_bar, color='purple', ls=':', lw=1.5, alpha=0.7, 
               label=f'Barrier at {phi_bar:.0f} GeV')
ax.set_xlabel(r'$\phi$ [GeV]')
ax.set_ylabel(r'$V(\phi)/v^4$')
ax.set_title(r'Full view: 0 - 5000 GeV')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Zoomed near vev 100-1500 GeV
ax = axes[0, 1]
phi_zoom = np.linspace(100, 1500, 10000)
V_zoom = np.array([V_total(p) for p in phi_zoom])
ax.plot(phi_zoom, V_zoom/v**4, 'r-', lw=2, label='Full 1-loop potential')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axhline(V_at_v/v**4, color='orange', ls=':', lw=1, alpha=0.7,
           label=rf'$V(v) = {V_at_v/v**4:.3e}\,v^4$')
ax.axvline(v, color='green', ls=':', lw=1.5, alpha=0.7, label=f'v = {v:.0f} GeV')
if has_barrier:
    ax.axvline(phi_bar, color='purple', ls=':', lw=1.5, alpha=0.7,
               label=f'Barrier at {phi_bar:.0f} GeV')
ax.set_xlabel(r'$\phi$ [GeV]')
ax.set_ylabel(r'$V(\phi)/v^4$')
ax.set_title('Zoomed: vev + barrier region')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Extreme zoom on the well + barrier
ax = axes[1, 0]
lo_phi = 200
hi_phi = min(phi_bar*1.5, 2000) if has_barrier else 1000
phi_xz = np.linspace(lo_phi, hi_phi, 10000)
V_xz = np.array([V_total(p) for p in phi_xz])
ax.plot(phi_xz, V_xz/v**4, 'r-', lw=2.5, label='1-loop potential')
ax.axhline(V_at_v/v**4, color='orange', ls=':', lw=1, alpha=0.7)
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(v, color='green', ls='--', lw=1.5, alpha=0.5, label=f'v = {v:.0f} GeV')
if has_barrier:
    ax.axvline(phi_bar, color='purple', ls='--', lw=1.5, alpha=0.5, 
               label=f'Barrier top: {phi_bar:.0f} GeV')
    ax.axhline(V_bar/v**4, color='purple', ls=':', lw=1, alpha=0.5)
    dV = (V_bar - V_at_v)/v**4
    mid_phi = (v + phi_bar)/2
    mid_V = (V_at_v + V_bar)/(2*v**4)
    ax.annotate(
        f'Barrier height\n= {dV:.3e} $v^4$\n= {(V_bar-V_at_v):.2e} GeV$^4$',
        xy=(mid_phi, mid_V), fontsize=10, ha='center', color='darkred',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
ax.set_xlabel(r'$\phi$ [GeV]')
ax.set_ylabel(r'$V(\phi)/v^4$')
ax.set_title('Extreme zoom: Well + barrier')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel 4: Log-log view from 1 to 10^4 GeV
ax = axes[1, 1]
phi_log = np.logspace(0, 4, 5000)
V_log = np.array([V_total(p) for p in phi_log])
V_pos = np.where(V_log > 0, V_log, np.nan)
V_neg = np.where(V_log < 0, -V_log, np.nan)
ax.plot(phi_log, V_pos, 'r-', lw=2, label=r'$V > 0$')
ax.plot(phi_log, V_neg, 'r--', lw=2, label=r'$|V|$ (where $V < 0$)')
ax.axvline(v, color='green', ls=':', lw=1.5, alpha=0.7, label=f'v = {v:.0f} GeV')
if has_barrier:
    ax.axvline(phi_bar, color='purple', ls=':', lw=1.5, alpha=0.7,
               label=f'Barrier at {phi_bar:.0f} GeV')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\phi$ [GeV]')
ax.set_ylabel(r'$|V(\phi)|$ [GeV$^4$]')
ax.set_title(r'Log-log: $|V(\phi)|$ from 1 to $10^4$ GeV')
ax.set_xlim(1, 1e4)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/vev_well_corrected.png', dpi=200, bbox_inches='tight')
print(f"\nSaved to results/vev_well_corrected.png")
