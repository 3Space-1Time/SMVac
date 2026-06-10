#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "simplebounce.h" 

using namespace std;
namespace SB = simplebounce;

// SM Constants & RGE from previous solver
const double v = 246.22;
const double MPlanck = 1.22e19;
const double Mtau = 1.777;
const double Mb = 4.0;
const double alpha3_at_Mz = 0.1184; 
const double pi = 3.14159265358979323846;
const double PI2 = pi * pi;
const double LOOP1 = 16.0 * PI2;
const double LOOP2 = LOOP1 * LOOP1;
const double LOOP3 = LOOP2 * LOOP1;
const double g1init = 0.46;
const double g2init = 0.65;
const double g3init = 1.1666;

struct Params { double g1, g2, g3, yt, yb, ytau, lambda, phi = 0; };

class RGEHelper {
    vector<pair<double, Params>> table;
public:
    void clear() { table.clear(); }
    void add_point(double t, const Params& p) { table.push_back({t, p}); }
    Params get_params(double t) const {
        if (table.empty()) return {0};
        auto it = upper_bound(table.begin(), table.end(), t, 
            [](double val, const pair<double, Params>& elem){ return val < elem.first; });
        if (it == table.begin()) return table.front().second;
        if (it == table.end()) return table.back().second;
        const auto& p1 = *(it - 1);
        const auto& p2 = *it;
        double factor = (t - p1.first) / (p2.first - p1.first);
        Params result = p1.second;
        result.lambda = p1.second.lambda + factor * (p2.second.lambda - p1.second.lambda);
        return result;
    }
};

class SMPotential : public SB::GenericModel {
    RGEHelper& rge;
    double mu_inst;
public:
    SMPotential(RGEHelper& rge_ref, double mu) : rge(rge_ref), mu_inst(mu) { this->setNphi(1); }
    double vpot(const double* phi) const override {
        double h = phi[0] * mu_inst;
        if (h <= v) return 0; 
        double t = 2.0 * log(h); 
        Params p = rge.get_params(t);
        return 0.25 * p.lambda * pow(phi[0], 4); 
    }
    void calcDvdphi(const double* phi, double* dvdphi_out) const override {
        double h = phi[0] * mu_inst;
        if (h <= v) { dvdphi_out[0] = 0; return; }
        double t = 2.0 * log(h); 
        Params p = rge.get_params(t);
        dvdphi_out[0] = p.lambda * pow(phi[0], 3);
    }
    double get_energy_at(double phi_val) {
        double phi_arr[] = {phi_val};
        return vpot(phi_arr);
    }
};

// ... RGE Beta functions ...
double betaG1sq(const Params& p) { return p.g1*p.g1*p.g1*p.g1/LOOP1 * 4.1; } // Simplified for speed in this test setup
double betaG2sq(const Params& p) { return p.g2*p.g2*p.g2*p.g2/LOOP1 * -3.166; }
double betaG3sq(const Params& p) { return p.g3*p.g3*p.g3*p.g3/LOOP1 * -7; }
double betaLambda(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2, yt2=p.yt*p.yt;
    return (1/LOOP1) * (p.lambda*(12*p.lambda + 6*yt2 - 4.5*g2_2 - 0.9*g1_2) - 3*yt2*yt2 + 0.5625*g2_2*g2_2);
}
double betaYt2(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2, g3_2=p.g3*p.g3, yt2=p.yt*p.yt;
    return yt2/LOOP1 * (4.5*yt2 - 8*g3_2 - 2.25*g2_2 - 0.85*g1_2);
}

Params rk4_single_step(const Params& y, double t, double dt) {
    auto dydt = [&](const Params& p) -> Params {
        Params dy; dy.g1 = 0.5/p.g1*betaG1sq(p); dy.g2 = 0.5/p.g2*betaG2sq(p); dy.g3 = 0.5/p.g3*betaG3sq(p);
        dy.yt = 0.5/p.yt*betaYt2(p); dy.lambda = betaLambda(p); return dy;
    };
    Params k1 = dydt(y);
    Params next_y = y;
    next_y.g1 += dt*k1.g1; next_y.g2 += dt*k1.g2; next_y.g3 += dt*k1.g3; 
    next_y.yt += dt*k1.yt; next_y.lambda += dt*k1.lambda;
    return next_y;
}

// Global physics setup for the chosen point
RGEHelper rge;
double mu_inst = MPlanck;
double phi_true_guess = -1.0;

void setup_physics() {
    double Mt = 173.1;
    double Mh = 125.1;
    double lambda0 = 0.12604 + 0.00206*(Mh - 125.15) - 0.00004*(Mt - 173.34);
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    double t = 2*log(172.5);
    Params y = {g1init, g2init, g3init, yt0, 0, 0, lambda0, 0};
    
    rge.add_point(t, y);
    bool is_unstable = false;
    double dt = 0.2;
    while (t < 2*log(MPlanck)) {
        y = rk4_single_step(y, t, dt);
        t += dt;
        rge.add_point(t, y);
        if (!is_unstable && y.lambda < 0) {
            mu_inst = exp(t/2);
            is_unstable = true;
        }
    }
    
    SMPotential pot(rge, mu_inst);
    double min_energy = 0.0;
    for (double p = 1.0; p < 10000.0; p *= 1.05) {
        double e = pot.get_energy_at(p);
        if (e < min_energy) { min_energy = e; phi_true_guess = p; }
    }
}

int main() {
    setup_physics();
    SMPotential pot(rge, mu_inst);
    
    cout << "--- Discretization Error Test ---" << endl;
    ofstream f_disc("../../results/discretization_error.csv");
    f_disc << "N,S_exact\n";
    
    // Test N from 100 to 6400
    vector<int> N_vals = {50, 100, 200, 400, 800, 1600, 3200, 6400};
    for (int N : N_vals) {
        SB::BounceCalculator bounce;
        bounce.setModel(&pot);
        double phi_true_arr[] = {phi_true_guess}; 
        double phi_false_arr[] = {0.0};
        bounce.setVacuum(phi_true_arr, phi_false_arr);
        bounce.setMaxN(10000); // Allow it to solve without aborting
        bounce.setN(N);
        bounce.solve();
        f_disc << N << "," << bounce.action() << "\n";
        cout << "N = " << N << " -> S = " << bounce.action() << endl;
    }
    f_disc.close();
    
    cout << "--- Finite Volume Error Test ---" << endl;
    ofstream f_vol("../../results/finite_volume_error.csv");
    f_vol << "Rmax,S_exact\n";
    
    // Fix dr = 1.0 / 1000 = 0.001
    double dr = 0.001;
    vector<double> R_vals = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0};
    for (double R : R_vals) {
        int N = max(10, (int)(R / dr));
        SB::BounceCalculator bounce;
        bounce.setModel(&pot);
        double phi_true_arr[] = {phi_true_guess}; 
        double phi_false_arr[] = {0.0};
        bounce.setVacuum(phi_true_arr, phi_false_arr);
        bounce.setMaxN(10000);
        bounce.setRmax(R);
        bounce.setN(N);
        bounce.solve();
        f_vol << R << "," << bounce.action() << "\n";
        cout << "Rmax = " << R << " (N=" << N << ") -> S = " << bounce.action() << endl;
    }
    f_vol.close();
    
    return 0;
}
