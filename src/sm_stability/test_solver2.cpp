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

#include <omp.h>

using namespace std;

// Alias for convenience

// --- CONSTANTS ---
const double v = 246.22;
const double MPlanck = 1.22e19;
const double Mtau = 1.777;
const double Mb = 4.0;
const double alpha3_at_Mz = 0.1184; 
const double pi = M_PI;
const double pi2 = pow(4*pi, 2);
const double pi4 = pi2*pi2;
const double pi6 = pi2*pi2*pi2;

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

// --- FULL 3-LOOP BETA FUNCTIONS ---
double betaG1sq(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2, g3_2=p.g3*p.g3;
    double g1_4=g1_2*g1_2, g2_4=g2_2*g2_2, g3_4=g3_2*g3_2;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;
    
    double t1 = g1_4/pi2 * 4.1;
    double t2 = g1_4/pi4 * (4.4*g3_2 + 2.7*g2_2 + 3.98*g1_2 - 1.7*yt2 - 0.5*yb2 - 1.5*ytau2);
    double t3 = g1_4/pi6 * (
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

    double t1 = g2_4/pi2 * (-3.166);
    double t2 = g2_4/pi4 * (12*g3_2 + 5.83*g2_2 + 0.9*g1_2 - 1.5*yt2 - 1.5*yb2 - 0.5*ytau2);
    double t3 = g2_4/pi6 * (
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

    double t1 = -g3_4/pi2 * 7;
    double t2 = g3_4/pi4 * (-26*g3_2 + 4.5*g2_2 + 1.1*g1_2 - 2*yt2 - 2*yb2);
    double t3 = g3_4/pi6 * (
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

    double t1 = (1/pi2) * (p.lambda*(12*p.lambda + 6*yt2 + 6*yb2 + 2*ytau2 - 4.5*g2_2 - 0.9*g1_2) 
                - 3*yt4 - 3*pow(p.yb,4) - pow(p.ytau,4) + 0.5625*g2_4 + 0.0675*g1_4 + 0.225*g2_2*g1_2);
    
    double t2 = (1/pi4) * (
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
    
    double t1 = yt2/pi2 * (4.5*yt2 + 1.5*yb2 + ytau2 - 8*g3_2 - 2.25*g2_2 - 0.85*g1_2);
    double t2 = yt2/pi4 * (
        yt2*(-12*yt2 - 2.75*yb2 - 2.25*ytau2 - 12*p.lambda + 36*g3_2 + 14.06*g2_2 + 4.9*g1_2) +
        6*p.lambda*p.lambda - 108*pow(p.g3,4) - 5.75*pow(p.g2,4) + 1.9*pow(p.g1,4) + 
        9*g3_2*g2_2 + 1.2*g3_2*g1_2 - 0.45*g2_2*g1_2
    );
    return t1 + t2;
}

double betaYb2(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2, g3_2=p.g3*p.g3;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;
    return yb2/pi2 * (1.5*yt2 + 4.5*yb2 + ytau2 - 8*g3_2 - 2.25*g2_2 - 0.25*g1_2);
}

double betaYtau2(const Params& p) {
    double g1_2=p.g1*p.g1, g2_2=p.g2*p.g2;
    double yt2=p.yt*p.yt, yb2=p.yb*p.yb, ytau2=p.ytau*p.ytau;
    return ytau2/pi2 * (3*yt2 + 3*yb2 + 2.5*ytau2 - 2.25*g2_2 - 2.25*g1_2);
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
    const double TOL = 1e-10;
    while(true) {
        Params y1 = rk4_single_step(y, t, dt);
        Params y_half = rk4_single_step(y, t, dt/2);
        Params y2 = rk4_single_step(y_half, t + dt/2, dt/2);
        
        double error = std::abs(y1.lambda - y2.lambda) + std::abs(y1.yt - y2.yt);
        if (!std::isfinite(error) || !std::isfinite(y2.lambda) || !std::isfinite(y2.yt)) {
            return false; // Diverged / hit a pole
        }
        if (error < TOL) {
            y = y2;
            t += dt;
            if (error < TOL/10) dt *= 1.5;
            return true;
        } else if (dt < 1e-5) {
            return false; // Diverged / hit a pole
        } else {
            dt *= 0.5;
        }
    }
}


double evaluate_action_at_R(RGEHelper& rge, double mu_inst, double R) {
    double mu_R = mu_inst / R;
    if (mu_R <= 0) return 1e100;
    
    double t_R = 2.0 * log(mu_R);
    Params p_R = rge.get_params(t_R);
    double lambda_R = p_R.lambda;
    
    if (lambda_R >= 0) return 1e100; 
    
    double prefactor = sqrt(2.0 / std::abs(lambda_R));
    double kinetic_term = (16.0 * pi2) / (3.0 * std::abs(lambda_R));
    
    double potential_integral = 0;
    int N = 6400; // Fully converged per scaling test
    double x_min = -50.0;
    double x_max = 50.0;
    double dx = (x_max - x_min) / N;
    
    for (int i = 0; i <= N; ++i) {
        double x = x_min + i * dx;
        double e2x = exp(2.0*x);
        double e4x = e2x * e2x;
        
        double phi_x = prefactor * 2.0 / (R * (e2x + 1.0));
        
        double h = phi_x * mu_inst;
        double V_x = 0;
        if (h > v) {
            double t = 2.0 * log(h);
            Params p_phi = rge.get_params(t);
            V_x = 0.25 * p_phi.lambda * pow(phi_x, 4);
        }
        
        double integrand = 2.0 * pi2 * pow(R, 4) * e4x * V_x;
        
        double weight;
        if (i == 0 || i == N) weight = 1.0 / 3.0;
        else if (i % 2 == 1) weight = 4.0 / 3.0;
        else weight = 2.0 / 3.0;
        
        potential_integral += weight * integrand * dx;
    }
    
    return kinetic_term + potential_integral;
}

double find_minimum_action(RGEHelper& rge, double mu_inst, double t_min_lambda) {
    const double invphi = (sqrt(5.0) - 1.0) / 2.0;
    const double invphi2 = (3.0 - sqrt(5.0)) / 2.0;
    
    double mu_opt = exp(t_min_lambda / 2.0);
    double R_opt = mu_inst / mu_opt;
    double logR_opt = log(R_opt);
    double a = logR_opt - 8.0;  
    double b = logR_opt + 8.0;  
    double tol = 1e-10;
    
    double h = b - a;
    if (h <= tol) return evaluate_action_at_R(rge, mu_inst, exp(0.5*(a+b)));
    
    // N steps of Golden Section Search
    int n = int(ceil(log(tol / h) / log(invphi)));
    
    double c = a + invphi2 * h;
    double d = a + invphi * h;
    
    double yc = evaluate_action_at_R(rge, mu_inst, exp(c));
    double yd = evaluate_action_at_R(rge, mu_inst, exp(d));
    
    for (int k = 0; k < n; ++k) {
        if (yc < yd) {
            b = d;
            d = c;
            yd = yc;
            h = invphi * h;
            c = a + invphi2 * h;
            yc = evaluate_action_at_R(rge, mu_inst, exp(c));
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi * h;
            d = a + invphi * h;
            yd = evaluate_action_at_R(rge, mu_inst, exp(d));
        }
    }
    
    double min_R = exp(0.5 * (a + b));
    return evaluate_action_at_R(rge, mu_inst, min_R);
}

// --- CLASSIFICATION ---
std::tuple<int, double, double> classify_stability(double Mh, double Mt) {
    double lambda0 = Mh*Mh / (2*v*v);
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    
    double t0 = 2*log(172.5);
    double tPlanck = 2*log(MPlanck);
    
    Params y = {g1init, g2init, g3init, yt0, sqrt(2.0)*Mb/v, sqrt(2.0)*Mtau/v, lambda0, 0};
    
    RGEHelper rge;
    bool is_unstable = false;
    double mu_inst = MPlanck;
    
    double min_lambda = 0.0;
    double t_min_lambda = t0;
    
    double t = t0;
    double dt = 0.1;
    rge.add_point(t, y);
    
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        if (!rk4_adaptive_step(y, t, dt)) {
            return std::make_tuple(4, -1.0, -1.0); 
        }
        rge.add_point(t, y);
        
        if (std::abs(y.lambda) > 4*pi || y.yt > 4*pi) return std::make_tuple(4, -1.0, -1.0); 
        
        if (y.lambda < min_lambda) {
            min_lambda = y.lambda;
            t_min_lambda = t;
        }
        
        if (!is_unstable && y.lambda < 0) {
            mu_inst = exp(t/2.0);
            is_unstable = true;
        }
    }
    
    if (!is_unstable) return std::make_tuple(1, -1.0, -1.0); // Stable
    
    double S_approx = 8.0 * pi2 / (3.0 * std::abs(min_lambda));
    double S_exact = find_minimum_action(rge, mu_inst, t_min_lambda);
    
    if (S_exact > 400.0) return std::make_tuple(2, S_exact, S_approx); // Metastable
    return std::make_tuple(3, S_exact, S_approx); // Unstable
}



int main() {
    double Mh = 230.0;
    double Mt = 50.0;
    
    double lambda0 = Mh*Mh / (2*v*v);
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    double t0 = 2*log(172.5);
    double tPlanck = 2*log(MPlanck);
    
    Params y = {g1init, g2init, g3init, yt0, sqrt(2.0)*Mb/v, sqrt(2.0)*Mtau/v, lambda0, 0};
    
    double t = t0;
    double dt = 0.1;
    
    std::cout << "t0 = " << t0 << ", lambda0 = " << lambda0 << ", yt0 = " << yt0 << std::endl;
    
    int steps = 0;
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        if (!rk4_adaptive_step(y, t, dt)) {
            std::cout << "Pole hit (rk4 false) at t=" << t << ", lambda=" << y.lambda << std::endl;
            return 0;
        }
        steps++;
        if (steps % 100 == 0) std::cout << "t=" << t << ", lambda=" << y.lambda << ", yt=" << y.yt << ", g1=" << y.g1 << std::endl;
        
        if (std::abs(y.lambda) > 4*pi || y.yt > 4*pi) {
            std::cout << "Exceeded 4pi at t=" << t << ", lambda=" << y.lambda << std::endl;
            return 0;
        }
    }
    std::cout << "Reached tPlanck! lambda=" << y.lambda << std::endl;
    return 0;
}
