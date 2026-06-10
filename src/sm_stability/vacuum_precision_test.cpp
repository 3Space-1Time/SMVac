#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <atomic>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "simplebounce.h" 
#include <omp.h>

using namespace std;
namespace SB = simplebounce;

// --- CONSTANTS ---
const double v = 246.22;
const double MPlanck = 1.22e19;
const double Mtau = 1.777;
const double Mb = 4.0;
const double alpha3_at_Mz = 0.1184; 
const double pi = M_PI;
const double PI2 = pi * pi;
const double LOOP1 = 16.0 * PI2;
const double LOOP2 = LOOP1 * LOOP1;
const double LOOP3 = LOOP2 * LOOP1;

// SM Initial Couplings
const double g1init = 0.46;
const double g2init = 0.65;
const double g3init = 1.1666;

struct Params { 
    double g1, g2, g3, yt, yb, ytau, lambda, phi = 0; 
};

// --- RGE HELPER ---
class RGEHelper {
private:
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

// --- POTENTIAL DEFINITION ---
class SMPotential : public SB::GenericModel {
    RGEHelper& rge;
    double mu_inst;
public:
    SMPotential(RGEHelper& rge_ref, double mu) : rge(rge_ref), mu_inst(mu) { 
        this->setNphi(1); 
    }

    double vpot(const double* phi) const override {
        double h = phi[0] * mu_inst;
        if (h <= v) return 0; 
        double t = 2.0 * log(h); 
        Params p = rge.get_params(t);
        return 0.25 * p.lambda * pow(phi[0], 4); 
    }

    void calcDvdphi(const double* phi, double* dvdphi_out) const override {
        double h = phi[0] * mu_inst;
        if (h <= v) { 
            dvdphi_out[0] = 0; 
            return; 
        }
        double t = 2.0 * log(h); 
        Params p = rge.get_params(t);
        dvdphi_out[0] = p.lambda * pow(phi[0], 3);
    }

    double get_energy_at(double phi_val) {
        double phi_arr[] = {phi_val};
        return vpot(phi_arr);
    }
};

// --- FULL 3-LOOP BETA FUNCTIONS ---
double betaG1sq(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2, g3_2=p.g3*p.g3;
    double g1_4=g1_2*g1_2, g2_4=g2_2*g2_2, g3_4=g3_2*g3_2;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;
    
    double t1 = g1_4/LOOP1 * 4.1;
    double t2 = g1_4/LOOP2 * (4.4*g3_2 + 2.7*g2_2 + 3.98*g1_2 - 1.7*yt2 - 0.5*yb2 - 1.5*ytau2);
    double t3 = g1_4/LOOP3 * (
        yt2*(11.8125*yt2 - 5.8*g3_2 - 14.71*g2_2 - 3.53*g1_2) +
        p.lambda*(-1.8*p.lambda + 0.9*g2_2 + 0.54*g1_2) +
        59.4*g3_4 + 12.32*g2_4 - 16.19*g1_4 - 0.6*g3_2*g2_2 - 1.82*g3_2*g1_2 + 0.76*g2_2*g1_2
    );
    return t1 + t2 + t3;
}

double betaG2sq(const Params& p) {
    double g2_2=p.g2*p.g2, g3_2=p.g3*p.g3, g1_2=p.g1*p.g1;
    double g2_4=g2_2*g2_2, g3_4=g3_2*g3_2, g1_4=g1_2*g1_2;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;

    double t1 = g2_4/LOOP1 * (-3.166);
    double t2 = g2_4/LOOP2 * (12*g3_2 + 5.83*g2_2 + 0.9*g1_2 - 1.5*yt2 - 1.5*yb2 - 0.5*ytau2);
    double t3 = g2_4/LOOP3 * (
        yt2*(9.18*yt2 - 7*g3_2 - 22.78*g2_2 - 3.7*g1_2) +
        p.lambda*(-3*p.lambda + 1.5*g2_2 + 0.3*g1_2) +
        81*g3_4 + 188.05*g2_4 - 3.49*g1_4 + 39*g3_2*g2_2 - 0.2*g3_2*g1_2 + 5.45*g2_2*g1_2
    );
    return t1 + t2 + t3;
}

double betaG3sq(const Params& p) {
    double g3_2=p.g3*p.g3, g2_2=p.g2*p.g2, g1_2=p.g1*p.g1;
    double g3_4=g3_2*g3_2, g2_4=g2_2*g2_2, g1_4=g1_2*g1_2;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb;

    double t1 = -g3_4/LOOP1 * 7;
    double t2 = g3_4/LOOP2 * (-26*g3_2 + 4.5*g2_2 + 1.1*g1_2 - 2*yt2 - 2*yb2);
    double t3 = g3_4/LOOP3 * (
        yt2*(15*yt2 - 40*g3_2 - 11.625*g2_2 - 2.525*g1_2) +
        32.5*g3_4 + 13.625*g2_4 - 4.35*g1_4 + 21*g3_2*g2_2 + 5.13*g3_2*g1_2 - 0.075*g2_2*g1_2
    );
    return t1 + t2 + t3;
}

double betaLambda(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2, g3_2=p.g3*p.g3;
    double g1_4=g1_2*g1_2, g2_4=g2_2*g2_2, g3_4=g3_2*g3_2;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;
    double yt4=yt2*yt2;

    double t1 = (1/LOOP1) * (p.lambda*(12*p.lambda + 6*yt2 + 6*yb2 + 2*ytau2 - 4.5*g2_2 - 0.9*g1_2) 
                - 3*yt4 - 3*pow(p.yb,4) - pow(p.ytau,4) + 0.5625*g2_4 + 0.0675*g1_4 + 0.225*g2_2*g1_2);
    
    double t2 = (1/LOOP2) * (
        p.lambda*p.lambda*(-156*p.lambda - 72*yt2 - 72*yb2 - 24*ytau2 + 54*g2_2 + 10.8*g1_2) +
        p.lambda*yt2*(-1.5*yt2 - 21*yb2 + 40*g3_2 + 11.25*g2_2 + 4.25*g1_2) +
        p.lambda*(-4.5*g2_4 + 4.7*g1_4 + 2.9*g2_2*g1_2) + 
        yt4*(15*yt2 - 16*g3_2 - 0.8*g1_2) + yt2*(-1.125*g2_4 - 0.85*g1_4 + 3.15*g2_2*g1_2)
    );
    return t1 + t2;
}

double betaYt2(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2, g3_2=p.g3*p.g3;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;
    
    double t1 = yt2/LOOP1 * (4.5*yt2 + 1.5*yb2 + ytau2 - 8*g3_2 - 2.25*g2_2 - 0.85*g1_2);
    double t2 = yt2/LOOP2 * (
        yt2*(-12*yt2 - 2.75*yb2 - 2.25*ytau2 - 12*p.lambda + 36*g3_2 + 14.06*g2_2 + 4.9*g1_2) +
        6*p.lambda*p.lambda - 108*pow(p.g3,4) - 5.75*pow(p.g2,4) + 1.9*pow(p.g1,4) + 
        9*g3_2*g2_2 + 1.2*g3_2*g1_2 - 0.45*g2_2*g1_2
    );
    return t1 + t2;
}

double betaYb2(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2, g3_2=p.g3*p.g3;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;
    return yb2/LOOP1 * (1.5*yt2 + 4.5*yb2 + ytau2 - 8*g3_2 - 2.25*g2_2 - 0.25*g1_2);
}

double betaYtau2(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;
    return ytau2/LOOP1 * (3*yt2 + 3*yb2 + 2.5*ytau2 - 2.25*g2_2 - 2.25*g1_2);
}

// --- RK4 SOLVER ---
Params rk4_single_step(const Params& y, double t, double dt) {
    auto dydt = [&](const Params& p) -> Params {
        Params dy;
        dy.g1 = 0.5/p.g1*betaG1sq(p); dy.g2 = 0.5/p.g2*betaG2sq(p); dy.g3 = 0.5/p.g3*betaG3sq(p);
        dy.yt = 0.5/p.yt*betaYt2(p); dy.yb = 0.5/p.yb*betaYb2(p); dy.ytau = 0.5/p.ytau*betaYtau2(p);
        dy.lambda = betaLambda(p); dy.phi = 0;
        return dy;
    };
    Params k1 = dydt(y);
    Params k2 = dydt({y.g1+0.5*dt*k1.g1, y.g2+0.5*dt*k1.g2, y.g3+0.5*dt*k1.g3, 
                      y.yt+0.5*dt*k1.yt, y.yb+0.5*dt*k1.yb, y.ytau+0.5*dt*k1.ytau, y.lambda+0.5*dt*k1.lambda, 0});
    Params k3 = dydt({y.g1+0.5*dt*k2.g1, y.g2+0.5*dt*k2.g2, y.g3+0.5*dt*k2.g3, 
                      y.yt+0.5*dt*k2.yt, y.yb+0.5*dt*k2.yb, y.ytau+0.5*dt*k2.ytau, y.lambda+0.5*dt*k2.lambda, 0});
    Params k4 = dydt({y.g1+dt*k3.g1, y.g2+dt*k3.g2, y.g3+dt*k3.g3, 
                      y.yt+dt*k3.yt, y.yb+dt*k3.yb, y.ytau+dt*k3.ytau, y.lambda+dt*k3.lambda, 0});
    
    Params next_y = y;
    next_y.g1+=dt/6*(k1.g1+2*k2.g1+2*k3.g1+k4.g1); next_y.g2+=dt/6*(k1.g2+2*k2.g2+2*k3.g2+k4.g2);
    next_y.g3+=dt/6*(k1.g3+2*k2.g3+2*k3.g3+k4.g3); next_y.yt+=dt/6*(k1.yt+2*k2.yt+2*k3.yt+k4.yt);
    next_y.yb+=dt/6*(k1.yb+2*k2.yb+2*k3.yb+k4.yb); next_y.ytau+=dt/6*(k1.ytau+2*k2.ytau+2*k3.ytau+k4.ytau);
    next_y.lambda+=dt/6*(k1.lambda+2*k2.lambda+2*k3.lambda+k4.lambda);
    return next_y;
}

bool rk4_adaptive_step(Params& y, double& t, double& dt) {
    const double TOL = 1e-6;
    while(true) {
        Params y1 = rk4_single_step(y, t, dt);
        Params y_half = rk4_single_step(y, t, dt/2);
        Params y2 = rk4_single_step(y_half, t + dt/2, dt/2);
        
        double error = std::abs(y1.lambda - y2.lambda) + std::abs(y1.yt - y2.yt);
        if (error < TOL || !std::isfinite(error)) {
            y = y2;
            t += dt;
            if (error < TOL/10) dt *= 1.5;
            return true;
        } else if (dt < 1e-5) {
            return false; 
        } else {
            dt *= 0.5;
        }
    }
}

// Returns tuple(S_exact, S_approx)
tuple<double, double> calculate_precision(double Mh, double Mt) {
    double lambda0 = 0.12604 + 0.00206*(Mh - 125.15) - 0.00004*(Mt - 173.34);
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    
    double t0 = 2*log(172.5);
    double tPlanck = 2*log(MPlanck);
    
    Params y = {g1init, g2init, g3init, yt0, sqrt(2)*Mb/v, sqrt(2)*Mtau/v, lambda0, 0};
    
    RGEHelper rge;
    double mu_inst = MPlanck;
    double lambda_min = 1.0; 
    
    double t = t0;
    double dt = 0.1;
    rge.add_point(t, y);
    bool is_unstable = false;
    
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        if (!rk4_adaptive_step(y, t, dt)) {
            return make_tuple(-1.0, -1.0); // Invalid / singularity
        }
        rge.add_point(t, y);
        
        if (y.lambda < lambda_min) {
            lambda_min = y.lambda;
        }
        
        if (std::abs(y.lambda) > 4*pi || y.yt > 4*pi) return make_tuple(-1.0, -1.0); 
        if (!is_unstable && y.lambda < 0) {
            mu_inst = exp(t/2);
            is_unstable = true;
        }
    }
    
    if (lambda_min >= 0) return make_tuple(-1.0, -1.0); // Stable, no bounce action
    
    // Theoretical Approximation: S ≈ 8π^2 / (3|λ_min|)
    double S_approx = (8.0 * PI2) / (3.0 * std::abs(lambda_min));
    
    SMPotential pot(rge, mu_inst);
    
    // Fine-grained logarithmic scan for SimpleBounce Vacuum
    double phi_true_guess = -1.0;
    double min_energy = 0.0;
    
    for (double p = 1.0; p < 10000.0; p *= 1.05) {
        double e = pot.get_energy_at(p);
        if (e < min_energy) { 
            min_energy = e;
            phi_true_guess = p;
        }
    }
    
    if (phi_true_guess < 0) {
        return make_tuple(-1.0, S_approx); // Couldn't find vacuum, fallback
    }

    SB::BounceCalculator bounce;
    bounce.setModel(&pot);
    
    double phi_true_arr[] = {phi_true_guess}; 
    double phi_false_arr[] = {0.0};
    bounce.setVacuum(phi_true_arr, phi_false_arr);
    
    int status = bounce.solve();
    if (status != 0) return make_tuple(-1.0, S_approx); // SimpleBounce failed
    
    double S_exact = bounce.action();
    return make_tuple(S_exact, S_approx);
}

void process_chunk(const vector<pair<double,double>>& points, 
                   vector<tuple<double,double,double,double>>& results, 
                   int start, int end) {
    std::atomic<int> completed(0);
    int total = end - start;
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = start; i < end; ++i) {
        double Mt = points[i].first;
        double Mh = points[i].second;
        auto precision = calculate_precision(Mh, Mt);
        results[i] = make_tuple(Mt, Mh, std::get<0>(precision), std::get<1>(precision));
        
        int done = ++completed;
        if (done % 30 == 0 || done == total) {
            #pragma omp critical
            {
                double pct = 100.0 * done / total;
                cout << "\r[";
                for (int p = 0; p < 50; ++p) {
                    if (p < (done * 50 / total)) cout << "=";
                    else cout << " ";
                }
                cout << "] " << fixed << setprecision(1) << pct << "% (" << done << "/" << total << ")" << flush;
            }
        }
    }
    cout << endl;
}

int main() {
    vector<pair<double,double>> points;
    
    // Dense grid around SM Vacuum
    // Mh in [124.0, 126.0], Mt in [172.0, 174.0], step 0.025
    for (double Mt = 172.0; Mt <= 174.0; Mt += 0.025) {
        for (double Mh = 124.0; Mh <= 126.0; Mh += 0.025) {
            points.emplace_back(Mt, Mh);
        }
    }
    
    cout << "Precision Mode: Analyzing " << points.size() << " points near SM vacuum..." << endl;
    
    vector<tuple<double,double,double,double>> results(points.size());
    process_chunk(points, results, 0, points.size());
    
    ofstream file("../../results/precision_data.csv");
    file << "Mt,Mh,S_exact,S_approx\n";
    for (size_t i = 0; i < results.size(); ++i) {
        double Mt = std::get<0>(results[i]);
        double Mh = std::get<1>(results[i]);
        double Sexact = std::get<2>(results[i]);
        double Sapprox = std::get<3>(results[i]);
        file << Mt << "," << Mh << "," << Sexact << "," << Sapprox << "\n";
    }
    file.close();
    cout << "Done. Saved to precision_data.csv" << endl;
    return 0;
}
