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
LOOP1 = 16.0 * PI2

Mt = 100.0
Mh = 5.0

# NNLO matching
lam_tree = Mh**2 / (2.0*v**2)
yt_Mt = np.sqrt(2.0)*Mt/v - 0.0587
d_lambda = -0.00313 - 0.00004*(Mt-173.34)
lam_Mt = lam_tree + d_lambda
g1 = 0.46266 + 0.00006*(Mt-173.34)
g2 = 0.65355 + 0.00002*(Mt-173.34)

print(f'lambda_tree = {lam_tree:.6f}')
print(f'lambda(Mt)  = {lam_Mt:.6f}')
print(f'yt(Mt)      = {yt_Mt:.6f}')

mu2 = lam_Mt * v**2
print(f'mu^2 = lam*v^2 = {mu2:.4f} GeV^2')

# 1-loop CW potential
def V_full(phi):
    V_tree = -mu2/2.0 * phi**2 + lam_Mt/4.0 * phi**4
    Mt2 = 0.5 * yt_Mt**2 * phi**2
    Mw2 = 0.25 * g2**2 * phi**2
    Mz2 = 0.25 * (g1**2 + g2**2) * phi**2
    mu_ren2 = Mt**2
    V_1loop = 0.0
    if Mt2 > 0:
        V_1loop += -12.0 * Mt2**2 * (np.log(Mt2/mu_ren2) - 1.5)
    if Mw2 > 0:
        V_1loop += 6.0 * Mw2**2 * (np.log(Mw2/mu_ren2) - 5.0/6.0)
    if Mz2 > 0:
        V_1loop += 3.0 * Mz2**2 * (np.log(Mz2/mu_ren2) - 5.0/6.0)
    V_1loop /= (64.0 * PI2)
    return V_tree + V_1loop

def V_tree_only(phi):
    return -mu2/2.0 * phi**2 + lam_Mt/4.0 * phi**4

# Find local min and barrier
phi_fine = np.linspace(1, 800, 100000)
V_vals = np.array([V_full(p) for p in phi_fine])

# Local min near v
half = len(phi_fine)//2
idx_min = np.argmin(V_vals[:half])
phi_min = phi_fine[idx_min]
V_min = V_vals[idx_min]

# Barrier: local max after the minimum
search_start = idx_min
search_end = min(idx_min + half, len(V_vals))
# Find where derivative changes sign (max after min)
dV = np.diff(V_vals[search_start:search_end])
# Find first sign change from positive to negative (peak)
sign_changes = np.where(np.diff(np.sign(dV)))[0]
if len(sign_changes) > 0:
    idx_bar = search_start + sign_changes[0] + 1
    phi_bar = phi_fine[idx_bar]
    V_bar = V_vals[idx_bar]
    has_barrier = True
else:
    has_barrier = False
    phi_bar = phi_min + 100
    V_bar = V_full(phi_bar)

print(f'Local minimum at phi = {phi_min:.1f} GeV, V = {V_min:.4e} GeV^4')
print(f'V(v=246) = {V_full(v):.4e} GeV^4')
if has_barrier:
    print(f'Barrier top at phi = {phi_bar:.1f} GeV, V = {V_bar:.4e} GeV^4')
    print(f'Barrier height = {V_bar - V_min:.4e} GeV^4')
    print(f'Barrier height / v^4 = {(V_bar - V_min)/v**4:.6e}')
else:
    print('No barrier found - potential monotonically decreases after minimum')

# ── PLOT ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    rf'Zoomed View of the EW Vacuum Well:  $M_t = 100$ GeV, $M_h = 5$ GeV,  '
    rf'$\lambda(M_t) = {lam_Mt:.1e}$',
    fontsize=14, fontweight='bold')

# Panel 1: Full view 0-2000 GeV
ax = axes[0]
phi_wide = np.linspace(1, 2000, 5000)
V_tree_w = np.array([V_tree_only(p) for p in phi_wide])
V_full_w = np.array([V_full(p) for p in phi_wide])
ax.plot(phi_wide, V_tree_w/v**4, 'b-', lw=1.5, alpha=0.5, label='Tree-level')
ax.plot(phi_wide, V_full_w/v**4, 'r-', lw=2, label='1-loop')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(v, color='green', ls=':', lw=1.5, alpha=0.7, label=f'v = {v:.0f} GeV')
ax.set_xlabel(r'$\phi$ [GeV]')
ax.set_ylabel(r'$V(\phi)/v^4$')
ax.set_title('Full view (0 - 2000 GeV)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Zoomed near the minimum 50-600 GeV
ax = axes[1]
phi_zoom = np.linspace(50, 600, 10000)
V_full_z = np.array([V_full(p) for p in phi_zoom])
V_tree_z = np.array([V_tree_only(p) for p in phi_zoom])
ax.plot(phi_zoom, V_tree_z/v**4, 'b-', lw=1.5, alpha=0.5, label='Tree-level')
ax.plot(phi_zoom, V_full_z/v**4, 'r-', lw=2, label='1-loop')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(v, color='green', ls=':', lw=1.5, alpha=0.7, label=f'v = {v:.0f} GeV')
ax.axvline(phi_min, color='orange', ls=':', lw=1.5, label=f'Min at {phi_min:.0f} GeV')
if has_barrier:
    ax.axvline(phi_bar, color='purple', ls=':', lw=1.5, label=f'Barrier at {phi_bar:.0f} GeV')
ax.set_xlabel(r'$\phi$ [GeV]')
ax.set_ylabel(r'$V(\phi)/v^4$')
ax.set_title('Zoomed: EW minimum + barrier region')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Extreme zoom on the well & barrier
ax = axes[2]
lo = max(phi_min*0.3, 10)
hi = phi_bar*1.3 if has_barrier else phi_min*3
phi_xz = np.linspace(lo, hi, 10000)
V_full_xz = np.array([V_full(p) for p in phi_xz])
ax.plot(phi_xz, V_full_xz/v**4, 'r-', lw=2.5, label='1-loop potential')
ax.axhline(V_min/v**4, color='orange', ls=':', lw=1, alpha=0.7,
           label=f'V(min) = {V_min/v**4:.2e} $v^4$')
if has_barrier:
    ax.axhline(V_bar/v**4, color='purple', ls=':', lw=1, alpha=0.7,
               label=f'V(barrier) = {V_bar/v**4:.2e} $v^4$')
    dV = (V_bar - V_min)/v**4
    ax.annotate(f'Barrier height\n= {dV:.2e} $v^4$',
                xy=((phi_min+phi_bar)/2, (V_min+V_bar)/(2*v**4)),
                fontsize=10, ha='center', color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(phi_min, color='orange', ls='--', lw=1, alpha=0.5)
if has_barrier:
    ax.axvline(phi_bar, color='purple', ls='--', lw=1, alpha=0.5)
ax.set_xlabel(r'$\phi$ [GeV]')
ax.set_ylabel(r'$V(\phi)/v^4$')
ax.set_title('Extreme zoom: Well depth & barrier')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('results/vev_well_zoom.png', dpi=200, bbox_inches='tight')
print('\nSaved to results/vev_well_zoom.png')
