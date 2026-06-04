"""
Physically correct implementation with proper inflationary dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from tqdm import tqdm

class CorrectedHiggsInflatonModel:
    """
    Physically motivated model with proper inflationary dynamics
    """
    
    def __init__(self):
        # Planck mass in GeV
        self.Mpl = 2.435e18  # GeV
        
        # Model parameters in natural units (GeV)
        self.params = {
            'm_phi': 1e13,      # Inflaton mass ~ 10^13 GeV (typical GUT scale)
            'v': 246,           # Higgs VEV in GeV
            'lam': 0.13,        # Higgs quartic
            'g': 1e-6,          # Coupling constant
            'H_inf': 1e13,      # Hubble during inflation (~GUT scale)
        }
        
        # Compute derived parameters
        self.params['phi_c'] = self.params['v'] * np.sqrt(self.params['lam']/self.params['g'])
        
    def potential(self, phi, h):
        """Scalar potential in GeV^4"""
        # Inflaton (quadratic)
        V_phi = 0.5 * self.params['m_phi']**2 * phi**2
        
        # Higgs (Mexican hat)
        V_h = 0.25 * self.params['lam'] * (h**2 - self.params['v']**2)**2
        
        # Interaction
        V_int = 0.5 * self.params['g'] * phi**2 * h**2
        
        return V_phi + V_h + V_int
    
    def hubble_rate(self, phi, h):
        """Hubble rate H in GeV"""
        V = self.potential(phi, h)
        return np.sqrt(V / (3 * self.Mpl**2))
    
    def slow_roll_parameters(self, phi, h):
        """Compute slow-roll parameters epsilon and eta"""
        V = self.potential(phi, h)
        
        # Numerical derivatives
        dphi = 1e-3 * phi if phi != 0 else 1e-3
        dh = 1e-3 * h if h != 0 else 1e-3
        
        V_phi_plus = self.potential(phi + dphi, h)
        V_phi_minus = self.potential(phi - dphi, h)
        dV_dphi = (V_phi_plus - V_phi_minus) / (2 * dphi)
        
        V_h_plus = self.potential(phi, h + dh)
        V_h_minus = self.potential(phi, h - dh)
        dV_dh = (V_h_plus - V_h_minus) / (2 * dh)
        
        # Second derivatives for eta
        V_phi_plus2 = self.potential(phi + 2*dphi, h)
        V_phi_minus2 = self.potential(phi - 2*dphi, h)
        d2V_dphi2 = (V_phi_plus - 2*V + V_phi_minus) / (dphi**2)
        
        epsilon_phi = 0.5 * (self.Mpl * dV_dphi / V)**2
        eta_phi = self.Mpl**2 * d2V_dphi2 / V
        
        return epsilon_phi, eta_phi
    
    def compute_efolds(self, phi_initial, h_initial):
        """
        Compute realistic number of e-folds during slow-roll.
        Returns when slow-roll conditions break (epsilon=1).
        """
        # Initial conditions
        phi = phi_initial
        h = h_initial
        
        # Track evolution
        N = 0.0
        max_steps = 10000
        dN = 0.01  # Small step in e-folds
        
        for step in range(max_steps):
            # Compute Hubble and derivatives
            H = self.hubble_rate(phi, h)
            V = self.potential(phi, h)
            
            # Numerical derivatives
            dphi = 1e-3 * phi if phi != 0 else 1e-3
            dh = 1e-3 * h if h != 0 else 1e-3
            
            dV_dphi = (self.potential(phi + dphi, h) - self.potential(phi - dphi, h)) / (2 * dphi)
            dV_dh = (self.potential(phi, h + dh) - self.potential(phi, h - dh)) / (2 * dh)
            
            # Slow-roll equations
            dphi_dN = -self.Mpl**2 * dV_dphi / V
            dh_dN = -self.Mpl**2 * dV_dh / V
            
            # Update fields
            phi += dphi_dN * dN
            h += dh_dN * dN
            
            # Ensure positive
            phi = max(phi, 0)
            
            # Update e-folds
            N += dN
            
            # Check slow-roll condition
            epsilon, eta = self.slow_roll_parameters(phi, h)
            if epsilon >= 1.0 or phi <= 0:
                break
        
        return N, phi, h
    
    def compute_density_perturbations(self, phi, h):
        """
        Compute amplitude of scalar perturbations Q using standard formula.
        Q ≈ H^2/(2π|dϕ/dt|) ≈ (1/√(24π²)) * V^{3/2}/(M_pl^3 |V'|)
        """
        V = self.potential(phi, h)
        
        # Compute derivative numerically
        dphi = 1e-3 * phi if phi != 0 else 1e-3
        dV_dphi = (self.potential(phi + dphi, h) - self.potential(phi - dphi, h)) / (2 * dphi)
        
        if abs(dV_dphi) < 1e-30:
            return 0.0
        
        # Standard formula for scalar perturbations
        Q = (1.0 / np.sqrt(24 * np.pi**2)) * (V**1.5) / (self.Mpl**3 * abs(dV_dphi))
        
        return Q
    
    def compute_euclidean_action(self, phi_false, h_false, phi_true, h_true):
        """
        Compute Euclidean action for tunneling.
        Simplified but physically motivated formula.
        """
        # Energy densities
        V_false = self.potential(phi_false, h_false)
        V_true = self.potential(phi_true, h_true)
        
        if V_false <= V_true:
            return 1e10  # No tunneling possible
        
        # Energy difference
        epsilon = V_false - V_true
        
        # Characteristic energy scale
        V_scale = max(V_false, V_true)
        
        # Simplified Euclidean action formula
        # S_E ~ M_pl^4 / ε for small bubbles
        # We add Higgs suppression when V_Higgs becomes significant
        
        # Higgs energy contribution
        V_H_false = 0.25 * self.params['lam'] * (h_false**2 - self.params['v']**2)**2
        V_H_true = 0.25 * self.params['lam'] * (h_true**2 - self.params['v']**2)**2
        
        # Base tunneling action (without Higgs effects)
        S_E_base = self.Mpl**4 / epsilon
        
        # Higgs suppression factor
        # When Higgs potential is significant, tunneling is suppressed
        if V_H_false > 0 or V_H_true > 0:
            # Suppression factor grows when Higgs energy is comparable to inflaton energy
            V_phi_false = 0.5 * self.params['m_phi']**2 * phi_false**2
            V_phi_true = 0.5 * self.params['m_phi']**2 * phi_true**2
            
            if V_phi_false > 0:
                higgs_ratio_false = V_H_false / V_phi_false
            else:
                higgs_ratio_false = 1e10  # Large suppression
            
            if V_phi_true > 0:
                higgs_ratio_true = V_H_true / V_phi_true
            else:
                higgs_ratio_true = 1e10
            
            # Total suppression factor
            suppression = 1 + 1e3 * (higgs_ratio_false + higgs_ratio_true)
            
            S_E = S_E_base * suppression
        else:
            S_E = S_E_base
        
        # Ensure S_E is positive and not too small
        S_E = max(S_E, 1.0)
        
        return S_E
    
    def transition_probability(self, phi_false, h_false, phi_true, h_true):
        """
        Compute transition probability in stationary measure.
        P ∝ exp(-S_E) * exp(3N)
        """
        # Euclidean action
        S_E = self.compute_euclidean_action(phi_false, h_false, phi_true, h_true)
        
        # Number of e-folds
        N, phi_end, h_end = self.compute_efolds(phi_true, h_true)
        
        # Log probability (up to normalization)
        log_P = -S_E + 3*N
        
        # Compute density perturbations
        Q = self.compute_density_perturbations(phi_true, h_true)
        
        return {
            'phi_false': phi_false,
            'phi_true': phi_true,
            'h_false': h_false,
            'h_true': h_true,
            'S_E': S_E,
            'N': N,
            'log_P': log_P,
            'P': np.exp(log_P) if log_P < 700 else np.exp(700),  # Avoid overflow
            'Q': Q,
            'log_Q': np.log10(Q) if Q > 0 else -50,
            'phi_end': phi_end,
            'h_end': h_end
        }
    
    def parameter_scan(self):
        """Run physically meaningful parameter scan"""
        results = []
        
        # Scan over false vacuum phi values
        # From above critical point to well below
        phi_c = self.params['phi_c']
        
        # Create phi points: some above phi_c, some below
        phi_points_above = np.logspace(np.log10(phi_c * 10), np.log10(phi_c * 1000), 30)
        phi_points_below = np.logspace(np.log10(phi_c / 100), np.log10(phi_c / 1.1), 30)
        phi_points = np.concatenate([phi_points_above, phi_points_below])
        phi_points.sort()
        
        print(f"\nScanning {len(phi_points)} points from {phi_points[0]:.2e} to {phi_points[-1]:.2e} GeV")
        print(f"Critical phi (Higgs phase transition): {phi_c:.2e} GeV")
        
        for phi_false in tqdm(phi_points, desc="Parameter scan"):
            # Determine Higgs state
            if phi_false >= phi_c:
                h_false = 0.0  # Symmetric phase
                h_true = 0.0   # Tunneling within symmetric phase
            else:
                # Broken phase: Higgs at minimum
                h_false = np.sqrt(self.params['v']**2 - 
                                (self.params['g']/self.params['lam']) * phi_false**2)
                h_true = h_false  # Tunneling within broken phase
            
            # Assume phi decreases slightly after tunneling
            phi_true = 0.95 * phi_false
            
            # Compute probability
            result = self.transition_probability(phi_false, h_false, phi_true, h_true)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_results(self, df):
        """Analyze and display results"""
        print("\n" + "="*70)
        print("ANALYSIS RESULTS")
        print("="*70)
        
        # Basic statistics
        print(f"\nNumber of configurations: {len(df)}")
        print(f"Range of φ_false: {df['phi_false'].min():.2e} to {df['phi_false'].max():.2e} GeV")
        print(f"Range of N (e-folds): {df['N'].min():.2f} to {df['N'].max():.2f}")
        print(f"Range of Q: {df['Q'].min():.2e} to {df['Q'].max():.2e}")
        print(f"Range of log(P): {df['log_P'].min():.2f} to {df['log_P'].max():.2f}")
        
        # Find most probable configuration
        if len(df) > 0:
            idx_max = df['log_P'].idxmax()
            best = df.loc[idx_max]
            
            print("\n" + "="*70)
            print("MOST PROBABLE CONFIGURATION:")
            print(f"  φ_false: {best['phi_false']:.2e} GeV")
            print(f"  φ_true: {best['phi_true']:.2e} GeV")
            print(f"  Higgs in true vacuum: {best['h_true']:.2e} GeV")
            print(f"  Euclidean action S_E: {best['S_E']:.2e}")
            print(f"  e-folds N: {best['N']:.2f}")
            print(f"  Density perturbation Q: {best['Q']:.2e}")
            print(f"  log probability: {best['log_P']:.2f}")
            
            # Check if Q is in observable range
            Q_obs_min = 1e-6
            Q_obs_max = 1e-4
            Q_best = best['Q']
            
            if Q_obs_min <= Q_best <= Q_obs_max:
                print(f"\n✓ Q IN OBSERVABLE RANGE ({Q_obs_min:.0e} to {Q_obs_max:.0e})")
            else:
                print(f"\n✗ Q OUTSIDE OBSERVABLE RANGE")
        
        # Count configurations with reasonable Q
        Q_reasonable = (df['Q'] >= 1e-10) & (df['Q'] <= 1e-4)
        N_reasonable = (df['N'] >= 40) & (df['N'] <= 70)
        
        n_good = (Q_reasonable & N_reasonable).sum()
        print(f"\nConfigurations with reasonable N (40-70) and Q (>1e-10):")
        print(f"  {n_good}/{len(df)} ({100*n_good/len(df):.1f}%)")
        
        if n_good > 0:
            # Average properties of good configurations
            good_df = df[Q_reasonable & N_reasonable]
            avg_logP = good_df['log_P'].mean()
            max_logP = good_df['log_P'].max()
            print(f"  Average log(P): {avg_logP:.2f}")
            print(f"  Maximum log(P): {max_logP:.2f}")
            
            # Check if good configurations are competitive
            if max_logP >= df['log_P'].max() * 0.9:  # Within 10% of maximum
                print("  ✓ Good configurations are probabilistically competitive")
            else:
                print("  ✗ Good configurations are suppressed")
        
        print("="*70)
        
        return df
    
    def plot_results(self, df):
        """Create informative plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Color by phase (above/below phi_c)
        phi_c = self.params['phi_c']
        colors = ['blue' if phi > phi_c else 'red' for phi in df['phi_false']]
        
        # 1. Euclidean action vs phi
        axes[0, 0].scatter(df['phi_false'], df['S_E'], c=colors, alpha=0.6, s=30)
        axes[0, 0].axvline(x=phi_c, color='black', linestyle='--', alpha=0.5, 
                          label=f'φ_c = {phi_c:.1e} GeV')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlabel(r'$\phi_{\text{false}}$ (GeV)')
        axes[0, 0].set_ylabel(r'$S_E$ (Euclidean action)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_title('Tunneling Suppression by Higgs')
        
        # 2. Number of e-folds vs phi
        axes[0, 1].scatter(df['phi_false'], df['N'], c=colors, alpha=0.6, s=30)
        axes[0, 1].axvline(x=phi_c, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=50, color='green', linestyle='-', alpha=0.5, 
                          label='Typical N ~ 50-60')
        axes[0, 1].axhline(y=60, color='green', linestyle='-', alpha=0.5)
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_xlabel(r'$\phi_{\text{false}}$ (GeV)')
        axes[0, 1].set_ylabel(r'$N$ (e-folds)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_title('Duration of Inflation')
        
        # 3. Probability vs phi
        axes[0, 2].scatter(df['phi_false'], df['log_P'], c=colors, alpha=0.6, s=30)
        axes[0, 2].axvline(x=phi_c, color='black', linestyle='--', alpha=0.5)
        axes[0, 2].set_xscale('log')
        axes[0, 2].set_xlabel(r'$\phi_{\text{false}}$ (GeV)')
        axes[0, 2].set_ylabel(r'$\log P$')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_title('Transition Probability')
        
        # 4. Q vs phi
        axes[1, 0].scatter(df['phi_false'], df['log_Q'], c=colors, alpha=0.6, s=30)
        axes[1, 0].axvline(x=phi_c, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=np.log10(2e-5), color='green', linestyle='-', 
                          alpha=0.7, label='Observed Q ≈ 2×10⁻⁵')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlabel(r'$\phi_{\text{false}}$ (GeV)')
        axes[1, 0].set_ylabel(r'$\log_{10} Q$')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_title('Density Perturbations')
        
        # 5. Probability vs Q
        scatter = axes[1, 1].scatter(df['log_Q'], df['log_P'], 
                                    c=np.log10(df['phi_false']),
                                    cmap='viridis', s=50, alpha=0.7)
        axes[1, 1].axvline(x=np.log10(2e-5), color='green', linestyle='-',
                          alpha=0.7, label='Observed Q')
        axes[1, 1].set_xlabel(r'$\log_{10} Q$')
        axes[1, 1].set_ylabel(r'$\log P$')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_title('Q-Catastrophe Check')
        plt.colorbar(scatter, ax=axes[1, 1], label=r'$\log_{10}(\phi_{\text{false}})$')
        
        # 6. S_E vs N tradeoff
        scatter2 = axes[1, 2].scatter(df['N'], np.log10(df['S_E']), 
                                     c=np.log10(df['phi_false']),
                                     cmap='plasma', s=50, alpha=0.7)
        axes[1, 2].set_xlabel(r'$N$ (e-folds)')
        axes[1, 2].set_ylabel(r'$\log_{10} S_E$')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_title('Tunneling vs Inflation Tradeoff')
        plt.colorbar(scatter2, ax=axes[1, 2], label=r'$\log_{10}(\phi_{\text{false}})$')
        
        # Add phase labels
        fig.text(0.02, 0.98, 'Blue: Higgs symmetric phase (φ > φ_c)\nRed: Higgs broken phase (φ < φ_c)', 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('physically_correct_results.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main execution"""
    print("="*70)
    print("PHYSICALLY CORRECT HIGGS-INFLATON MODEL")
    print("Testing Q-catastrophe resolution")
    print("="*70)
    
    # Initialize model
    model = CorrectedHiggsInflatonModel()
    
    print("\nModel parameters (natural units):")
    print(f"  Planck mass M_pl: {model.Mpl:.2e} GeV")
    print(f"  Inflaton mass m_φ: {model.params['m_phi']:.2e} GeV")
    print(f"  Higgs VEV v: {model.params['v']:.2f} GeV")
    print(f"  Higgs quartic λ: {model.params['lam']:.3f}")
    print(f"  Coupling g: {model.params['g']:.2e}")
    print(f"  Critical φ_c: {model.params['phi_c']:.2e} GeV")
    
    # Run parameter scan
    print("\nRunning physically motivated parameter scan...")
    df = model.parameter_scan()
    
    # Analyze results
    df = model.analyze_results(df)
    
    # Plot results
    print("\nGenerating plots...")
    model.plot_results(df)
    
    # Save data
    df.to_csv('physically_correct_data.csv', index=False)
    print("\nData saved to 'physically_correct_data.csv'")
    
    # Theoretical interpretation
    print("\n" + "="*70)
    print("THEORETICAL INTERPRETATION")
    print("="*70)
    
    # Check if mechanism works
    if len(df) > 0:
        # Separate symmetric and broken phases
        phi_c = model.params['phi_c']
        symmetric = df[df['phi_false'] > phi_c]
        broken = df[df['phi_false'] < phi_c]
        
        if len(symmetric) > 0 and len(broken) > 0:
            # Compare probabilities
            avg_logP_sym = symmetric['log_P'].mean()
            avg_logP_brk = broken['log_P'].mean()
            
            print(f"\nAverage log(P) in symmetric phase (φ > φ_c): {avg_logP_sym:.2f}")
            print(f"Average log(P) in broken phase (φ < φ_c): {avg_logP_brk:.2f}")
            
            if avg_logP_sym > avg_logP_brk:
                print("✓ Symmetric phase favored (Higgs suppression works)")
            else:
                print("✗ Broken phase favored (Higgs suppression fails)")
            
            # Check Q values
            avg_Q_sym = symmetric['Q'].mean()
            avg_Q_brk = broken['Q'].mean()
            
            print(f"\nAverage Q in symmetric phase: {avg_Q_sym:.2e}")
            print(f"Average Q in broken phase: {avg_Q_brk:.2e}")
            
            # Check if symmetric phase gives reasonable Q
            if 1e-10 < avg_Q_sym < 1e-4:
                print("✓ Symmetric phase gives reasonable Q values")
            else:
                print("✗ Symmetric phase Q values problematic")
    
    print("\nKey mechanism check:")
    print("1. When φ > φ_c: Higgs at 0, tunneling relatively easy")
    print("2. When φ < φ_c: Higgs at nonzero VEV, tunneling suppressed")
    print("3. Q-catastrophe suppressed if tunneling to large-N (small-Q) vacua")
    print("   is suppressed by Higgs energy contribution to S_E")
    print("="*70)

if __name__ == "__main__":
    main()