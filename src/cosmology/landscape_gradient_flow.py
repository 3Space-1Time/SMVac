import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import floor
from scipy.optimize import minimize

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Physics Scales
Scale_mass = 5.0        # Mass scale (m)
Scale_lam  = 0.5        # Interaction scale (lambda)
Scale_int  = 2.0        # Random Interaction strength

# Simulation Parameters
N = 5                   # Number of scalar fields
# Dynamic step sizing based on interaction strength vs mass gap
Steps = floor(100000 * Scale_int / min(Scale_mass, Scale_lam)) 
if Steps < 100: Steps = 100 # Safety floor

Tolerance = 1e-16       # Stability threshold
Drift_Tol = 1e-5        # Max gradient norm for final verification

# ==========================================
# 2. GENERATE RANDOM LANDSCAPE
# ==========================================
np.random.seed(42) 

def symmetrize(T):
    if T.ndim == 2: return (T + T.T) / 2
    return T 

print(f"--- Landscape Scanner: N={N} ---")
print(f"Interaction Scale: {Scale_int} | Steps: {Steps}")

# Base Potential
m_sq = np.abs(np.random.uniform(0.5, 1.5, N) * Scale_mass**2)
lam  = np.abs(np.random.uniform(0.5, 1.5, N) * Scale_lam)

# Perturbation (The Landscape)
A1 = np.random.normal(0, Scale_int, N)
A2 = symmetrize(np.random.normal(0, Scale_int, (N, N)))
A3 = np.random.normal(0, Scale_int/5.0, (N, N, N))
A4 = np.random.normal(0, Scale_int/20.0, (N, N, N, N))

# Higgs Mass Params
B1 = np.random.normal(0, Scale_int, N)
B2 = symmetrize(np.random.normal(0, Scale_int, (N, N)))

# ==========================================
# 3. HELPER FUNCTIONS (For Relaxation)
# ==========================================
def get_V_total(s, coupling):
    # Base
    val = np.sum(-m_sq * s**2 + lam * s**4)
    # Interaction
    v_int = np.dot(A1, s)
    v_int += np.einsum('ij,i,j', A2, s, s)
    v_int += np.einsum('ijk,i,j,k', A3, s, s, s)
    v_int += np.einsum('ijkl,i,j,k,l', A4, s, s, s, s)
    return val + coupling * v_int

def get_Grad_total(s, coupling):
    g_base = -2 * m_sq * s + 4 * lam * s**3
    g_int  = A1 + 2*np.dot(A2, s) 
    g_int += 3*np.einsum('ijk,j,k->i', A3, s, s) 
    g_int += 4*np.einsum('ijkl,j,k,l->i', A4, s, s, s)
    return g_base + coupling * g_int

def get_Hessian_total(s, coupling):
    H_base = np.diag(-2 * m_sq + 12 * lam * s**2)
    H_int  = 2*A2 
    H_int += 3*(np.einsum('ijk,k->ij', A3, s) + np.einsum('ikj,k->ij', A3, s)) # Approx symm
    H_int += 12*np.einsum('ijkl,k,l->ij', A4, s, s)
    return H_base + coupling * H_int

# ==========================================
# 4. INITIALIZE MINIMA
# ==========================================
base_vevs = np.sqrt(m_sq / (2 * lam))
sign_combos = list(itertools.product([1, -1], repeat=N))

minima_tracker = []
for signs in sign_combos:
    pos = base_vevs * np.array(signs)
    minima_tracker.append( {"pos": pos, "alive": True, "id": signs} )

# ==========================================
# 5. ROBUST ADIABATIC FLOW LOOP
# ==========================================
dt = 1.0 / Steps

for step in range(1, Steps + 1):
    if step % 500 == 0: 
        alive_c = sum([1 for v in minima_tracker if v["alive"]])
        print(f"  Step {step}/{Steps} | Tracking: {alive_c}")
        
    current_coupling = step * dt
    
    for vacuum in minima_tracker:
        if not vacuum["alive"]: continue
        
        S = vacuum["pos"]
        
        # --- A. PREDICTOR (Euler Step) ---
        H_tot = get_Hessian_total(S, current_coupling)
        
        # Force from the ADDED slice only (dt * V_int)
        g_int_slice = A1 + 2*np.dot(A2, S) + 3*np.einsum('ijk,j,k->i', A3, S, S) + 4*np.einsum('ijkl,j,k,l->i', A4, S, S, S)
        force = -1 * (dt * g_int_slice)
        
        try:
            dS = np.linalg.solve(H_tot, force)
            S += dS
        except np.linalg.LinAlgError:
            # Singularity hit: Do not kill, flag for Relaxation
            pass 

        # --- B. CORRECTOR (Newton Polish) ---
        # Snap back to the exact bottom of the current potential
        for _ in range(2):
            g_tot = get_Grad_total(S, current_coupling)
            H_tot = get_Hessian_total(S, current_coupling)
            try:
                corr = np.linalg.solve(H_tot, -g_tot)
                S += corr
            except:
                break
        
        vacuum["pos"] = S # Update position
        
        # --- C. STABILITY CHECK & RELAXATION ---
        # Check if we are still a minimum or if we became a saddle point
        try:
            eigvals = np.linalg.eigvalsh(get_Hessian_total(S, current_coupling))
            min_eig = np.min(eigvals)
        except:
            min_eig = -1.0

        if min_eig < -1e-5:
            # SADDLE POINT DETECTED!
            # Instead of dying, let the field roll down to the new deep minimum
            res = minimize(
                fun=get_V_total, 
                x0=S, 
                args=(current_coupling,), 
                jac=get_Grad_total, 
                method='Newton-CG', 
                options={'xtol': 1e-5}
            )
            if res.success:
                vacuum["pos"] = res.x
            else:
                vacuum["alive"] = False # Truly lost (runaway direction)

# ==========================================
# 6. FINAL DATA GATHERING
# ==========================================
print("Flow Complete. Gathering Data...")
final_V = []
final_mu = []
survived = 0

for vacuum in minima_tracker:
    if not vacuum["alive"]: continue
    
    S = vacuum["pos"]
    survived += 1
    
    # Final values at full coupling (coupling=1.0)
    V_val = get_V_total(S, 1.0)
    
    # Higgs Mass mu^2 = B1*S + B2*S*S
    mu_val = np.dot(B1, S) + np.einsum('ij,i,j', B2, S, S)
    
    final_V.append(V_val)
    final_mu.append(mu_val)

# ==========================================
# 7. PLOTTING
# ==========================================
plt.figure(figsize=(10, 7))

# Scatter plot with color mapping
sc = plt.scatter(final_V, final_mu, c=final_mu, cmap='coolwarm', 
            edgecolor='k', alpha=0.8, s=60)

plt.axhline(0, color='black', linestyle='--', linewidth=2, label=r"Critical Line ($\mu^2=0$)")

# Labels
plt.title(f"Landscape Scanner (N={N})\nScale Int={Scale_int} | Active Minima: {survived}", fontsize=14)
plt.xlabel(r"Scalar Potential Depth ($V_{min}$)", fontsize=12)
plt.ylabel(r"Higgs Mass Parameter ($\mu^2_{eff}$)", fontsize=12)

plt.colorbar(sc, label=r"$\mu^2$ Magnitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()