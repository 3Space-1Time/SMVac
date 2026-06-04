#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "simplebounce.h" 

using namespace std;
namespace SB = simplebounce;

// A simple well-behaved potential: V(phi) = phi^2 / 2 - phi^3 / 3
// This avoids the 10^14 scale floating point explosions of the SM potential,
// allowing us to clearly and accurately measure the scaling of numerical 
// Discretization and Finite-Volume errors for the Bounce algorithm.
class SimpleTestModel : public SB::GenericModel {
public:
    SimpleTestModel() { this->setNphi(1); }
    double vpot(const double* phi) const override {
        return 0.5 * phi[0] * phi[0] - (1.0/3.0) * phi[0] * phi[0] * phi[0];
    }
    void calcDvdphi(const double* phi, double* dvdphi_out) const override {
        dvdphi_out[0] = phi[0] - phi[0] * phi[0];
    }
};

int main() {
    SimpleTestModel pot;
    double phi_true_arr[] = {2.0}; // True vacuum is at phi=2, V = 2 - 8/3 = -0.66
    double phi_false_arr[] = {0.0};
    
    cout << "--- Discretization Error Test ---" << endl;
    ofstream f_disc("../../results/discretization_error.csv");
    f_disc << "N,S_exact\n";
    
    vector<int> N_vals = {100, 200, 400, 800, 1600, 3200, 6400};
    for (int N : N_vals) {
        SB::BounceCalculator bounce;
        bounce.setModel(&pot);
        bounce.setVacuum(phi_true_arr, phi_false_arr);
        bounce.setMaxN(100000); 
        bounce.setN(N);
        bounce.solve();
        f_disc << N << "," << bounce.action() << "\n";
        cout << "N = " << N << " -> S = " << bounce.action() << endl;
    }
    f_disc.close();
    
    cout << "--- Finite Volume Error Test ---" << endl;
    ofstream f_vol("../../results/finite_volume_error.csv");
    f_vol << "Rmax,S_exact\n";
    
    double dr = 0.005; 
    vector<double> R_vals = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0};
    for (double R : R_vals) {
        int N = max(20, (int)(R / dr));
        SB::BounceCalculator bounce;
        bounce.setModel(&pot);
        bounce.setVacuum(phi_true_arr, phi_false_arr);
        bounce.setMaxN(100000);
        bounce.setRmax(R);
        bounce.setN(N);
        bounce.solve();
        f_vol << R << "," << bounce.action() << "\n";
        cout << "Rmax = " << R << " (N=" << N << ") -> S = " << bounce.action() << endl;
    }
    f_vol.close();
    
    return 0;
}
