import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

def run_verification():
    df = pd.read_csv('data/potential_verification_grid.csv')
    phi = df['phi'].values
    V_solver = df['V_solver'].values
    V_full = df['V_full'].values
    dV_solver = df['dV_solver'].values
    dV_full = df['dV_full'].values
    d2V_solver = df['d2V_solver'].values
    d2V_full = df['d2V_full'].values
    
    # Check for NaNs
    if np.any(np.isnan(V_full)): print("FAIL: V_full contains NaN")
    if np.any(np.isnan(V_solver)): print("FAIL: V_solver contains NaN")
    
    dV_num_solver = np.gradient(V_solver, phi)
    dV_num_full = np.gradient(V_full, phi)
    
    max_rel_err_solver = np.max(np.abs((dV_solver - dV_num_solver) / np.where(dV_solver == 0, 1e-100, dV_solver))[np.abs(dV_solver) > 1e-5])
    max_rel_err_full = np.max(np.abs((dV_full - dV_num_full) / np.where(dV_full == 0, 1e-100, dV_full))[np.abs(dV_full) > 1e-5])
    
    print(f"Max rel error (Solver V) analytic vs numeric dV/dphi: {max_rel_err_solver:.3e}")
    print(f"Max rel error (Full SM V) analytic vs numeric dV/dphi: {max_rel_err_full:.3e}")
    
    # Read stationary points
    sp = pd.read_csv('data/verify_stationary_points.csv')
    print("STATIONARY POINTS FOUND:")
    print(sp.to_string())
    
    # Generate overlay plot for the discrepancy
    # Plotting V_solver vs V_full around the EW minimum and barrier
    mask = (phi > 10) & (phi < 400)
    plt.figure(figsize=(10,6))
    plt.plot(phi[mask], V_full[mask], 'b-', label='Full Tree-level SM $V(\phi)$')
    plt.plot(phi[mask], V_solver[mask], 'r--', label='High-field approx (solver_numerical.cpp)')
    plt.axhline(0, color='k', linestyle=':')
    plt.xlabel("$\phi$ (GeV)")
    plt.ylabel("$V(\phi)$ (GeV$^4$)")
    plt.title("Discrepancy at Low Field: Missing EW Minimum in high-field approximation")
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/potential_discrepancy.png', dpi=300)
    plt.close()
    
    print("Verification complete. Discrepancy plot saved.")

if __name__ == '__main__':
    run_verification()
