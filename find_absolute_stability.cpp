#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <tuple>
#include <string>
#include <iomanip>

using namespace std;

// --- CONSTANTS ---
const double pi = 3.14159265358979323846;
const double v = 246.22;
const double MPlanck = 1.22e19;
const double Mb = 4.18;
const double Mtau = 1.776;
const double Mz = 91.1876;
const double g1init = 0.3583;
const double g2init = 0.6478;
const double g3init = 1.1666;
const double alpha3_at_Mz = 0.1184;

// --- RGE SYSTEM ---
struct Params {
    double g1, g2, g3, yt, yb, ytau, lambda, c6;
};

void rge_derivs(double t, const Params& y, Params& dydt) {
    double g1_2 = y.g1*y.g1;
    double g2_2 = y.g2*y.g2;
    double g3_2 = y.g3*y.g3;
    double yt_2 = y.yt*y.yt;
    
    double factor = 1.0 / (16.0 * pi * pi);
    
    dydt.g1 = factor * (41.0/10.0) * y.g1 * g1_2;
    dydt.g2 = factor * (-19.0/6.0) * y.g2 * g2_2;
    dydt.g3 = factor * (-7.0) * y.g3 * g3_2;
    
    dydt.yt = factor * y.yt * (9.0/2.0 * yt_2 - 17.0/20.0 * g1_2 - 9.0/4.0 * g2_2 - 8.0 * g3_2);
    dydt.yb = 0;
    dydt.ytau = 0;
    
    dydt.lambda = factor * (24.0 * y.lambda * y.lambda + 12.0 * y.lambda * yt_2 - 6.0 * yt_2 * yt_2 
                            - 3.0 * y.lambda * (3.0 * g2_2 + 3.0/5.0 * g1_2) 
                            + 3.0/8.0 * (2.0 * g2_2 * g2_2 + pow(g2_2 + 3.0/5.0 * g1_2, 2)));
                            
    dydt.c6 = 0; 
}

bool rk4_adaptive_step(Params& y, double& t, double& dt) {
    Params k1, k2, k3, k4, temp;
    rge_derivs(t, y, k1);
    
    temp = {y.g1 + 0.5*dt*k1.g1, y.g2 + 0.5*dt*k1.g2, y.g3 + 0.5*dt*k1.g3, y.yt + 0.5*dt*k1.yt, y.yb, y.ytau, y.lambda + 0.5*dt*k1.lambda, 0};
    rge_derivs(t + 0.5*dt, temp, k2);
    
    temp = {y.g1 + 0.5*dt*k2.g1, y.g2 + 0.5*dt*k2.g2, y.g3 + 0.5*dt*k2.g3, y.yt + 0.5*dt*k2.yt, y.yb, y.ytau, y.lambda + 0.5*dt*k2.lambda, 0};
    rge_derivs(t + 0.5*dt, temp, k3);
    
    temp = {y.g1 + dt*k3.g1, y.g2 + dt*k3.g2, y.g3 + dt*k3.g3, y.yt + dt*k3.yt, y.yb, y.ytau, y.lambda + dt*k3.lambda, 0};
    rge_derivs(t + dt, temp, k4);
    
    y.g1 += (dt/6.0)*(k1.g1 + 2*k2.g1 + 2*k3.g1 + k4.g1);
    y.g2 += (dt/6.0)*(k2.g2 + 2*k2.g2 + 2*k3.g2 + k4.g2);
    y.g3 += (dt/6.0)*(k3.g3 + 2*k2.g3 + 2*k3.g3 + k4.g3);
    y.yt += (dt/6.0)*(k1.yt + 2*k2.yt + 2*k3.yt + k4.yt);
    y.lambda += (dt/6.0)*(k1.lambda + 2*k2.lambda + 2*k3.lambda + k4.lambda);
    
    t += dt;
    return true;
}

double get_Mh_calc(double lambda0, double yt0) {
    return sqrt(2.0 * lambda0) * v; 
}

bool is_stable(double Mh, double Mt) {
    double lambda0 = Mh*Mh / (2*v*v);
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    
    double t0 = 2*log(172.5);
    double tPlanck = 2*log(MPlanck);
    
    Params y = {g1init, g2init, g3init, yt0, sqrt(2.0)*Mb/v, sqrt(2.0)*Mtau/v, lambda0, 0};
    
    double t = t0;
    double dt = 0.1;
    
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        rk4_adaptive_step(y, t, dt);
        
        if (y.lambda < 0.0) return false; // Instability triggered
    }
    
    return true; // Absolute Stability
}

int main() {
    ofstream file("results/absolute_stability_boundary.csv");
    file << "Mt,Mh_stable\n";
    
    for (double Mt = 100.0; Mt <= 250.0; Mt += 2.0) {
        double low = 10.0;
        double high = 300.0;
        
        for (int i=0; i<40; ++i) {
            double mid = 0.5 * (low + high);
            if (is_stable(mid, Mt)) {
                high = mid; // If stable, boundary is lower
            } else {
                low = mid;  // If unstable, boundary is higher
            }
        }
        
        double mh_bound = 0.5 * (low + high);
        
        // Output calculated physical Mh
        double lambda0 = mh_bound*mh_bound / (2*v*v);
        double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
        double Mh_calc = get_Mh_calc(lambda0, yt0);
        
        file << Mt << "," << Mh_calc << "\n";
        cout << "Mt: " << Mt << " | Mh_stable: " << Mh_calc << endl;
    }
    
    file.close();
    return 0;
}
