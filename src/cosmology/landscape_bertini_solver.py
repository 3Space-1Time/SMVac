import numpy as np
import os, subprocess
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N = 10                   # Number of scalar fields
Run_Solver = True       # Set to True to calculate, False to just plot existing data
Cmd = "../../external/bertini/bertini"     # Executable name
import os
if os.name == 'nt' or os.path.exists(Cmd + ".exe"): Cmd += ".exe"

# Physics Scales
M_scale = 5.0
L_scale = 0.5
I_scale = 0.2

np.random.seed(42)

# ==========================================
# PART 1: GENERATE LANDSCAPE
# ==========================================
print(f"--- 1. Generating Landscape (N={N}) ---")

# Mass and Interaction Terms
m2 = np.abs(np.random.uniform(0.5, 1.5, N) * M_scale**2)
lam = np.abs(np.random.uniform(0.5, 1.5, N) * L_scale)

# Interaction Coefficients (A) and Higgs Couplings (B)
A1 = np.random.normal(0, I_scale, N)
A2 = (np.random.normal(0, I_scale, (N,N)) + np.random.normal(0, I_scale, (N,N)).T)/2
A3 = np.random.normal(0, I_scale/5, (N,N,N))
A4 = np.random.normal(0, I_scale/20, (N,N,N,N))

B1 = np.random.normal(0, I_scale, N)
B2 = (np.random.normal(0, I_scale, (N,N)) + np.random.normal(0, I_scale, (N,N)).T)/2

# Create Equations string: dV/dxi = 0
eqs = []
vars = ",".join([f"x{i}" for i in range(N)])

for i in range(N):
    eq = f"-2*{m2[i]:.4f}*x{i} + 4*{lam[i]:.4f}*x{i}^3"
    
    if abs(A1[i]) > 1e-5: eq += f"+{A1[i]:.4f}"
    
    for j in range(N):
        if abs(A2[i,j]) > 1e-5: eq += f"+{2*A2[i,j]:.4f}*x{j}"
        
    for j in range(N):
        for k in range(N):
            val = 3 * (A3[i,j,k] + A3[i,k,j])/2
            if abs(val) > 1e-5: eq += f"+{val:.4f}*x{j}*x{k}"
            
    for j in range(N):
        for k in range(N):
            for l in range(N):
                if abs(A4[i,j,k,l]) > 1e-5: eq += f"+{4*A4[i,j,k,l]:.4f}*x{j}*x{k}*x{l}"
    
    eqs.append(eq)

with open("input", "w") as f:
    # TrackType: 0 is faster (total degree) and produces cleaner output files
    f.write("CONFIG\nTrackType: 0;\nEND;\n\nINPUT\n")
    f.write(f"variable_group {vars};\n")
    f.write(f"function {','.join([f'f{i}' for i in range(N)])};\n")
    for i in range(N): f.write(f"f{i} = {eqs[i]};\n")
    f.write("END;\n")

# ==========================================
# PART 2: RUN SOLVER
# ==========================================
if Run_Solver:
    print("--- 2. Running Bertini ---")
    if not os.path.exists(Cmd):
        # Fallback for Linux users
        if os.path.exists("./bertini"): Cmd = "./bertini"
        else:
            print("Error: bertini executable not found.")
            exit()
            
    subprocess.run([Cmd, "input"], check=True)

# ==========================================
# PART 3: ANALYZE RESULTS
# ==========================================
print("--- 3. Analyzing Results ---")

# Try to find the best output file
target = None
for f in ["real_finite_solutions", "finite_solutions", "main_data"]:
    if os.path.exists(f):
        target = f
        break

if not target:
    print("Error: No output files found.")
    exit()

print(f"Reading from '{target}'...")

real_points = []
with open(target, 'r') as f:
    content = f.read()

# --- Universal Parser ---
# Handles both clean lists (TrackType 0) and messy blocks (TrackType 1)
is_clean_format = ("Path number" not in content) and ("Variables:" not in content)

lines = content.split('\n')
current_sol = []

for line in lines:
    parts = line.strip().split()
    
    # We look for lines with exactly 2 numbers (Real Imag)
    if len(parts) == 2:
        try:
            re = float(parts[0])
            im = float(parts[1])
            
            # Save real part if imaginary part is small
            if abs(im) < 1e-6:
                current_sol.append(re)
            else:
                # If one component is complex, discard the whole solution (reset buffer)
                # But for robustness, we mark it invalid and wait for block end
                current_sol.append(None) 
                
        except ValueError:
            continue

    # Check if we finished reading one solution
    if len(current_sol) == N:
        # Verify all components were real numbers (no Nones)
        if all(x is not None for x in current_sol):
            real_points.append(np.array(current_sol))
        current_sol = [] # Reset for next

print(f"Parsed {len(real_points)} valid real solutions.")

# --- CHECK STABILITY ---
V_vals, mu_vals = [], []
stable_count = 0

for S in real_points:
    # Hessian
    H = np.diag(-2*m2 + 12*lam*S**2)
    H += 2*A2
    H += 6 * np.einsum('ijk,k->ij', A3, S)
    H += 12 * np.einsum('ijkl,k,l->ij', A4, S, S)
    
    # Stability Check (Min Eigenvalue > 0)
    if np.min(np.linalg.eigvalsh(H)) > 0:
        stable_count += 1
        
        # Potentials
        v = np.sum(-m2*S**2 + lam*S**4)
        v += np.dot(A1,S) + np.einsum('ij,i,j',A2,S,S) + \
             np.einsum('ijk,i,j,k',A3,S,S,S) + np.einsum('ijkl,i,j,k,l',A4,S,S,S,S)
        
        mu = np.dot(B1,S) + np.einsum('ij,i,j',B2,S,S)
        
        V_vals.append(v)
        mu_vals.append(mu)

print(f"Found {stable_count} Stable Minima.")

# ==========================================
# PART 4: PLOT
# ==========================================
if stable_count > 0:
    plt.figure(figsize=(8,6))
    plt.scatter(V_vals, mu_vals, c=mu_vals, cmap='coolwarm', edgecolors='k', s=80)
    plt.axhline(0, color='k', linestyle='--', label="Critical Line")
    plt.xlabel("Scalar Potential Energy (V)")
    plt.ylabel("Higgs Mass Parameter (mu^2)")
    plt.title(f"Landscape (N={N}) | {stable_count} Minima")
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig("landscape_plot.png", dpi=150)
    print("Success! Plot saved to 'landscape_plot.png'")
else:
    print("No stable minima found. Try lowering I_scale or checking N.")