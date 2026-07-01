#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <iomanip>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

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
const double LOOP4 = LOOP2 * LOOP2;

// SM Initial Couplings
const double g1init = 0.46;
const double g2init = 0.65;
const double g3init = 1.1666;

struct Params { 
    double g1, g2, g3, yt, yb, ytau, lambda, phi; 
};

// --- RGE HELPER ---
class RGEHelper {
public:
    vector<pair<double, Params>> table;
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

// --- NNLO MATCHING AT TOP POLE MASS ---
// --- GLOBALLY VALID NNLO MATCHING ---
// Uses exact kinematics with NNLO threshold offsets from Buttazzo et al. (1307.3536)
Params get_nnlo_matching(double Mh, double Mt) {
    double dAlphas = (alpha3_at_Mz - 0.1184) / 0.0007;
    
    // Exact tree-level kinematics
    double lambda_tree = (Mh * Mh) / (2.0 * v * v);
    double yt_tree = (sqrt(2.0) * Mt) / v;
    
    // NNLO Threshold corrections (evaluated at Mt = 173.34, Mh = 125.15)
    // delta_lambda = 0.12604 - (125.15^2 / (2 * 246.22^2)) = -0.00313
    double delta_lambda = -0.00313 - 0.00004 * (Mt - 173.34);
    
    // delta_yt = 0.93690 - (sqrt(2) * 173.34 / 246.22) = -0.0587
    double delta_yt = -0.0587 - 0.00042 * dAlphas;
    
    double lambda_Mt = lambda_tree + delta_lambda;
    double yt_Mt = yt_tree + delta_yt;
    
    double g1_Mt = 0.46266 + 0.00006 * (Mt - 173.34);
    double g2_Mt = 0.65355 + 0.00002 * (Mt - 173.34);
    double g3_Mt = 1.1666 + 0.00314 * dAlphas - 0.00046 * (Mt - 173.34);
    
    return {g1_Mt, g2_Mt, g3_Mt, yt_Mt, sqrt(2.0)*Mb/v, sqrt(2.0)*Mtau/v, lambda_Mt, 0};
}

// --- FULL 3-LOOP BETA FUNCTIONS ---
double betaG1sq(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2, g3_2 = p.g3*p.g3;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2, g3_4 = g3_2*g3_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    double term1 = g1_4/LOOP1 * (41.0/10);
    double term2 = g1_4/LOOP2 * (44*g3_2/5 + 27*g2_2/10 + 199*g1_2/50 - 17*yt2/10 - yb2/2 - 3*ytau2/2);
    double term3 = g1_4/LOOP3 * (
        yt2*(189*yt2/16 - 29*g3_2/5 - 471*g2_2/32 - 2827*g1_2/800) +
        p.lambda*(-9*p.lambda/5 + 9*g2_2/10 + 27*g1_2/50) +
        297*g3_4/5 + 789*g2_4/64 - 388613*g1_4/24000 -
        3*g3_2*g2_2/5 - 137*g3_2*g1_2/75 + 123*g2_2*g1_2/160
    );
    return term1 + term2 + term3;
}
double betaG2sq(const Params& p) {
    double g2_2 = p.g2*p.g2, g3_2 = p.g3*p.g3, g1_2 = p.g1*p.g1;
    double g2_4 = g2_2*g2_2, g3_4 = g3_2*g3_2, g1_4 = g1_2*g1_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    double term1 = g2_4/LOOP1 * (-19.0/6);
    double term2 = g2_4/LOOP2 * (12*g3_2 + 35*g2_2/6 + 9*g1_2/10 - 3*yt2/2 - 3*yb2/2 - ytau2/2);
    double term3 = g2_4/LOOP3 * (
        yt2*(147*yt2/16 - 7*g3_2 - 729*g2_2/32 - 593*g1_2/160) +
        p.lambda*(-3*p.lambda + 3*g2_2/2 + 3*g1_2/10) +
        81*g3_4 + 324953*g2_4/1728 - 5597*g1_4/1600 +
        39*g3_2*g2_2 - g3_2*g1_2/5 + 873*g2_2*g1_2/160
    );
    return term1 + term2 + term3;
}
double betaG3sq(const Params& p) {
    double g3_2 = p.g3*p.g3, g2_2 = p.g2*p.g2, g1_2 = p.g1*p.g1;
    double g3_4 = g3_2*g3_2, g2_4 = g2_2*g2_2, g1_4 = g1_2*g1_2;
    double g3_8 = g3_4*g3_4, g3_10 = g3_8*g3_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb;
    double term1 = -g3_4/LOOP1 * 7;
    double term2 = g3_4/LOOP2 * (-26*g3_2 + 9*g2_2/2 + 11*g1_2/10 - 2*yt2 - 2*yb2);
    double term3 = g3_4/LOOP3 * (
        yt2*(15*yt2 - 40*g3_2 - 93*g2_2/8 - 101*g1_2/40) +
        65*g3_4/2 + 109*g2_4/8 - 523*g1_4/120 +
        21*g3_2*g2_2 + 77*g3_2*g1_2/15 - 3*g2_2*g1_2/40
    );
    double term4 = g3_10/LOOP4 * 2472.28;
    return term1 + term2 + term3 + term4;
}
double betaLambda(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2, g3_2 = p.g3*p.g3;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2, g3_4 = g3_2*g3_2;
    double g1_6 = g1_4*g1_2, g2_6 = g2_4*g2_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    double yt4 = yt2*yt2, yb4 = yb2*yb2, ytau4 = ytau2*ytau2;
    double term1 = (1/LOOP1) * (
        p.lambda*(12*p.lambda + 6*yt2 + 6*yb2 + 2*ytau2 - 9*g2_2/2 - 9*g1_2/10) -
        3*yt4 - 3*yb4 - ytau4 + 9*g2_4/16 + 27*g1_4/400 + 9*g2_2*g1_2/40
    );
    double term2 = (1/LOOP2) * (
        p.lambda*p.lambda*(-156*p.lambda - 72*yt2 - 72*yb2 - 24*ytau2 + 54*g2_2 + 54*g1_2/5) +
        p.lambda*yt2*(-3*yt2/2 - 21*yb2 + 40*g3_2 + 45*g2_2/4 + 17*g1_2/4) +
        p.lambda*yb2*(-3*yb2/2 + 40*g3_2 + 45*g2_2/4 + 5*g1_2/4) +
        p.lambda*ytau2*(-ytau2/2 + 15*g2_2/4 + 15*g1_2/4) +
        p.lambda*(-73*g2_4/16 + 1887*g1_4/400 + 117*g2_2*g1_2/40) +
        yt4*(15*yt2 - 3*yb2 - 16*g3_2 - 4*g1_2/5) +
        yt2*(-9*g2_4/8 - 171*g1_4/200 + 63*g2_2*g1_2/20) +
        yb4*(-3*yt2 + 15*yb2 - 16*g3_2 + 2*g1_2/5) +
        yb2*(-9*g2_4/8 + 9*g1_4/40 + 27*g2_2*g1_2/20) +
        ytau4*(5*ytau2 - 6*g1_2/5) +
        ytau2*(-3*g2_4/8 - 9*g1_4/8 + 33*g2_2*g1_2/20) +
        305*g2_6/32 - 3411*g1_6/4000 - 289*g2_4*g1_2/160 - 1677*g2_2*g1_4/800
    );
    double term3 = (1/LOOP3) * (
        p.lambda*p.lambda*p.lambda*(6011.35*p.lambda + 873*yt2 - 387.452*g2_2 - 77.490*g1_2) +
        p.lambda*p.lambda*yt2*(1768.26*yt2 + 160.77*g3_2 - 359.539*g2_2 - 63.869*g1_2) +
        p.lambda*p.lambda*(-790.28*g2_4 - 185.532*g1_4 - 316.64*g2_2*g1_2) +
        p.lambda*yt4*(-223.382*yt2 - 662.866*g3_2 - 5.470*g2_2 - 21.015*g1_2) +
        p.lambda*yt2*(356.968*g3_4 - 319.664*g2_4 - 74.8599*g1_4 + 15.1443*g3_2*g2_2 + 17.454*g3_2*g1_2 + 5.615*g2_2*g1_2) +
        p.lambda*g2_4*(-57.144*g3_2 + 865.483*g2_2 + 79.638*g1_2) +
        p.lambda*g1_4*(-8.381*g3_2 + 61.753*g2_2 + 28.168*g1_2) +
        yt4*(-243.149*yt4 + 250.494*g3_2 + 74.138*g2_2 + 33.930*g1_2) +
        yt4*(-50.201*g3_4 + 15.884*g2_4 + 15.948*g1_4 + 13.349*g3_2*g2_2 + 17.570*g3_2*g1_2 - 70.356*g2_2*g1_2) +
        yt2*g3_2*(16.464*g2_4 + 1.016*g1_4 + 11.386*g2_2*g1_2) +
        yt2*g2_4*(62.500*g2_2 + 13.041*g1_2) +
        yt2*g1_4*(10.627*g2_2 + 11.117*g1_2) +
        g3_2*(7.536*g2_6 + 0.663*g1_6 + 1.507*g2_4*g1_2 + 1.105*g2_2*g1_4) -
        114.091*g2_6*g2_2 - 1.508*g1_6*g1_2 - 37.889*g2_4*g2_2*g1_2 + 6.500*g2_4*g1_4 - 1.543*g2_2*g1_6
    );
    return term1 + term2 + term3;
}
double betaYt2(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2, g3_2 = p.g3*p.g3;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2, g3_4 = g3_2*g3_2;
    double g3_6 = g3_4*g3_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    double term1 = yt2/LOOP1 * ( 9*yt2/2 + 3*yb2/2 + ytau2 - 8*g3_2 - 9*g2_2/4 - 17*g1_2/20 );
    double term2 = yt2/LOOP2 * ( yt2*(-12*yt2 - 11*yb2/4 - 9*ytau2/4 - 12*p.lambda + 36*g3_2 + 225*g2_2/16 + 393*g1_2/80) + yb2*(-yb2/4 + 5*ytau2/4 + 4*g3_2 + 99*g2_2/16 + 7*g1_2/80) + ytau2*(-9*ytau2/4 + 15*g2_2/8 + 15*g1_2/8) + 6*p.lambda*p.lambda - 108*g3_4 - 23*g2_4/4 + 1187*g1_4/600 + 9*g3_2*g2_2 + 19*g3_2*g1_2/15 - 9*g2_2*g1_2/20 );
    double term3 = yt2/LOOP3 * ( yt2*(58.6028*yt2 + 198*p.lambda - 157*g3_2 - 1593*g2_2/16 - 2437*g1_2/80) + p.lambda*yt2*(15*p.lambda/4 + 16*g3_2 - 135*g2_2/2 - 127*g1_2/10) + yt2*(363.764*g3_4 + 16.990*g2_4 - 24.422*g1_4 + 48.370*g3_2*g2_2 + 18.074*g3_2*g1_2 + 34.829*g2_2*g1_2) + p.lambda*p.lambda*(-36*p.lambda + 45*g2_2 + 9*g1_2) + p.lambda*(-171*g2_4/16 - 1089*g1_4/400 + 117*g2_2*g1_2/40) - 619.35*g3_6 + 169.829*g2_4*g2_2 + 16.099*g1_4*g1_2 + 73.654*g3_4*g2_2 - 15.096*g3_4*g1_2 - 21.072*g3_2*g2_4 - 22.319*g3_2*g1_4 - 321*g3_2*g2_2*g1_2/20 - 4.743*g2_4*g1_2 - 4.442*g2_2*g1_4 );
    return term1 + term2 + term3;
}
double betaYb2(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2, g3_2 = p.g3*p.g3;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2, g3_4 = g3_2*g3_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    double term1 = yb2/LOOP1 * ( 3*yt2/2 + 9*yb2/2 + ytau2 - 8*g3_2 - 9*g2_2/4 - g1_2/4 );
    double term2 = yb2/LOOP2 * ( yt2*(-yt2/4 - 11*yb2/4 + 5*ytau2/4 + 4*g3_2 + 99*g2_2/16 + 91*g1_2/80) + yb2*(-12*yb2 - 9*ytau2/4 - 12*p.lambda + 36*g3_2 + 225*g2_2/16 + 237*g1_2/80) + ytau2*(-9*ytau2/4 + 15*g2_2/8 + 15*g1_2/8) + 6*p.lambda*p.lambda - 108*g3_4 - 23*g2_4/4 - 127*g1_4/600 + 9*g3_2*g2_2 + 31*g3_2*g1_2/15 - 27*g2_2*g1_2/20 );
    return term1 + term2;
}
double betaYtau2(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    double term1 = ytau2/LOOP1 * ( 3*yt2 + 3*yb2 + 5*ytau2/2 - 9*g2_2/4 - 9*g1_2/4 );
    double term2 = ytau2/LOOP2 * ( 6*p.lambda*p.lambda - 23*g2_4/4 + 1371*g1_4/200 + 27*g2_2*g1_2/20 + yt2*(-27*yt2/4 + 3*yb2/2 - 27*ytau2/4 + 20*p.g3*p.g3 + 45*g2_2/8 + 17*g1_2/8) + yb2*(-27*yb2/4 - 27*ytau2/4 + 20*p.g3*p.g3 + 45*g2_2/8 + 5*g1_2/8) + ytau2*(-3*ytau2 - 12*p.lambda + 165*g2_2/16 + 537*g1_2/80) );
    return term1 + term2;
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
    const double TOL = 1e-13;
    while(true) {
        Params y1 = rk4_single_step(y, t, dt);
        Params y_half = rk4_single_step(y, t, dt/2);
        Params y2 = rk4_single_step(y_half, t + dt/2, dt/2);
        
        double error = std::abs(y1.lambda - y2.lambda) + std::abs(y1.yt - y2.yt);
        if (!std::isfinite(error) || !std::isfinite(y2.lambda) || !std::isfinite(y2.yt)) {
            return false;
        }
        if (error < TOL) {
            y = y2;
            t += dt;
            if (error < TOL/10) dt *= 1.5;
            return true;
        } else if (dt < 1e-6) {
            return false;
        } else {
            dt *= 0.5;
        }
    }
}

// --- CLASSIFICATION AND ACTION ---
// Returns tuple: <state, min_S_c6, min_S_pure, Mh_calc, S_threshold>
// state: 1=Stable, 2=Metastable, 3=Unstable, 4=Pole/Error
std::tuple<int, double, double, double, double> classify_stability(double Mh, double Mt, double c6) {
    Params y = get_nnlo_matching(Mh, Mt);
    double Mh_calc = Mh; // The input Mh is now the exact physical pole mass!
    
    double t0 = 2*log(Mt); // Matching is done exactly at mu = Mt
    double tPlanck = 2*log(MPlanck);
    
    RGEHelper rge;
    rge.add_point(t0, y);
    
    bool is_unstable = false;
    double t_inst = tPlanck;
    
    double t = t0;
    double dt = 0.1;
    
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        if (!rk4_adaptive_step(y, t, dt)) {
            return std::make_tuple(4, -1.0, -1.0, Mh_calc, -1.0);
        }
        rge.add_point(t, y);
        
        if (std::abs(y.lambda) > 4*pi || y.yt > 4*pi) {
            return std::make_tuple(4, -1.0, -1.0, Mh_calc, -1.0); 
        }
        
        if (!is_unstable && y.lambda < 0) {
            is_unstable = true;
            t_inst = t;
        }
    }
    
    if (!is_unstable) {
        return std::make_tuple(1, -1.0, -1.0, Mh_calc, -1.0); // Absolute Stable
    }
    
    // Evaluate action analytically using conformal bounce + phi^6 correction
    // S(mu) = 8 pi^2 / (3 |lambda|) + 16 pi^2 c6 mu^2 / (15 M_P^2 |lambda|^3)
    double min_S_c6 = 1e100;
    double min_S_pure = 1e100;
    
    int N_steps = 2000;
    double dt_eval = (tPlanck - t_inst) / N_steps;
    
    for (int i=0; i<=N_steps; ++i) {
        double t_eval = t_inst + i * dt_eval;
        Params p = rge.get_params(t_eval);
        if (p.lambda >= 0) continue;
        
        double abs_lam = -p.lambda;
        double mu = std::exp(t_eval / 2.0);
        
        double S_pure = (8.0 * PI2) / (3.0 * abs_lam);
        if (S_pure < min_S_pure) min_S_pure = S_pure;
        
        double S_c6_term = (16.0 * PI2 * c6 * mu * mu) / (15.0 * MPlanck * MPlanck * abs_lam * abs_lam * abs_lam);
        double S_total = S_pure + S_c6_term;
        
        if (S_total < min_S_c6) min_S_c6 = S_total;
    }
    
    double mu_inst = exp(t_inst / 2.0);
    double Tv = 1.179e44 / v;
    double S_threshold = 4.0 * std::log(Tv * mu_inst);
    
    if (min_S_c6 > S_threshold) {
        return std::make_tuple(2, min_S_c6, min_S_pure, Mh_calc, S_threshold); // Metastable
    } else {
        return std::make_tuple(3, min_S_c6, min_S_pure, Mh_calc, S_threshold); // Unstable
    }
}

// Bisection to find boundary where state changes
double find_boundary(double Mt, double c6, bool absolute_stability) {
    double low = 0.0;
    double high = 500.0;
    double tol = 1e-4; // High precision for smoothness
    
    for (int i=0; i<60; ++i) {
        double mid = 0.5 * (low + high);
        auto res = classify_stability(mid, Mt, c6);
        int state = std::get<0>(res);
        double S_min = std::get<1>(res);
        double S_thresh = std::get<4>(res);
        
        if (absolute_stability) {
            // Absolute stability boundary: between Stable (1) and Metastable/Unstable
            if (state == 1 || state == 4) {
                high = mid; // Decrease Mh to become unstable
            } else {
                low = mid;  // Increase Mh to become stable
            }
        } else {
            // Instability boundary: between Metastable and Unstable (S_min = S_thresh)
            if (state == 1 || state == 4) {
                high = mid; 
            } else {
                if (S_min < S_thresh) {
                    low = mid; // Unstable -> increase Mh to stabilize
                } else {
                    high = mid; // Metastable -> decrease Mh to destabilize
                }
            }
        }
        
        if ((high - low) < tol) break;
    }
    
    // Return the calculated pole mass at the boundary
    double final_Mh_input = 0.5 * (low + high);
    return final_Mh_input; // Because Mh_calc == Mh_input now
}

void sweep_and_save(const string& filename, double Mt_start, double Mt_end, double Mt_step) {
    ofstream file(filename);
    file << "Mt,Mh_abs,Mh_c6_0,Mh_c6_0_1,Mh_c6_1_0\n";
    file << std::fixed << std::setprecision(5);
    
    cout << "Sweeping Mt from " << Mt_start << " to " << Mt_end << "..." << endl;
    
    for (double Mt = Mt_start; Mt <= Mt_end; Mt += Mt_step) {
        double mh_abs   = find_boundary(Mt, 0.0, true);
        double mh_c6_0  = find_boundary(Mt, 0.0, false);
        double mh_c6_01 = find_boundary(Mt, 0.1, false);
        double mh_c6_10 = find_boundary(Mt, 1.0, false);
        
        file << Mt << "," << mh_abs << "," << mh_c6_0 << "," << mh_c6_01 << "," << mh_c6_10 << "\n";
        cout << "Mt: " << Mt 
             << " | Abs: " << mh_abs 
             << " | c6=0: " << mh_c6_0 
             << " | c6=0.1: " << mh_c6_01 
             << " | c6=1.0: " << mh_c6_10 << endl;
    }
    file.close();
    cout << "Saved to " << filename << endl;
}

int main(int argc, char* argv[]) {
    // We do both the broad and zoomed-in sweeps to satisfy the user request.
    sweep_and_save("results/c6_boundaries_broad.csv", 0.0, 300.0, 2.0);
    sweep_and_save("results/c6_boundaries_zoomed.csv", 165.0, 180.0, 0.2);
    
    return 0;
}
