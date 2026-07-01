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
    double g1, g2, g3, yt, yb, ytau, lambda, phi = 0; 
};

// --- RGE HELPER ---
class RGEHelper {
public:
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

// --- 1-LOOP EFFECTIVE POTENTIAL ---
double V1_loop(double phi, double yt) {
    double Mt2 = 0.5 * yt * yt * phi * phi;
    double Mw2 = 0.25 * g2init * g2init * phi * phi;
    double Mz2 = 0.25 * (g1init * g1init + g2init * g2init) * phi * phi;
    
    double mu = 173.34;
    double mu2 = mu * mu;
    
    double term_t = -12.0 * Mt2 * Mt2 * (std::log(Mt2 / mu2) - 1.5);
    double term_W = 6.0 * Mw2 * Mw2 * (std::log(Mw2 / mu2) - 5.0/6.0);
    double term_Z = 3.0 * Mz2 * Mz2 * (std::log(Mz2 / mu2) - 5.0/6.0);
    
    return (term_t + term_W + term_Z) / (64.0 * PI2);
}

double get_Mh_calc(double lambda0, double yt) {
    double h = 1e-4;
    double Vp = V1_loop(v + h, yt);
    double Vm = V1_loop(v - h, yt);
    double V0 = V1_loop(v, yt);
    
    double V1_prime = (Vp - Vm) / (2.0 * h);
    double V1_double_prime = (Vp - 2.0 * V0 + Vm) / (h * h);
    
    double Mh2 = 2.0 * lambda0 * v * v + V1_double_prime - (1.0 / v) * V1_prime;
    return std::sqrt(Mh2);
}

// --- POTENTIAL DEFINITION ---

// --- GLOBALLY VALID NNLO MATCHING ---
Params get_nnlo_matching(double Mh, double Mt) {
    double dAlphas = (alpha3_at_Mz - 0.1184) / 0.0007;
    double lambda_tree = (Mh * Mh) / (2.0 * v * v);
    double yt_tree = (sqrt(2.0) * Mt) / v;
    double delta_lambda = -0.00313 - 0.00004 * (Mt - 173.34);
    double delta_yt = -0.0587 - 0.00042 * dAlphas;
    double lambda_Mt = lambda_tree + delta_lambda;
    double yt_Mt = yt_tree + delta_yt;
    double g1_Mt = 0.46266 + 0.00006 * (Mt - 173.34);
    double g2_Mt = 0.65355 + 0.00002 * (Mt - 173.34);
    double g3_Mt = 1.1666 + 0.00314 * dAlphas - 0.00046 * (Mt - 173.34);
    return {g1_Mt, g2_Mt, g3_Mt, yt_Mt, sqrt(2.0)*Mb/v, sqrt(2.0)*Mtau/v, lambda_Mt, 0};
}

// --- EFFECTIVE COUPLING (HIGH SCALE POTENTIAL CORRECTIONS) ---
double get_lambda_eff(const Params& p) {
    double yt2 = p.yt * p.yt;
    double yt4 = yt2 * yt2;
    double term_t = -3.0 * yt4 * (std::log(0.5 * yt2) - 1.5);
    
    double g2_2 = p.g2 * p.g2;
    double g2_4 = g2_2 * g2_2;
    double term_W = 0.375 * g2_4 * (std::log(0.25 * g2_2) - 5.0/6.0);
    
    double g12 = p.g1 * p.g1 + g2_2;
    double g12_2 = g12 * g12;
    double term_Z = 0.1875 * g12_2 * (std::log(0.25 * g12) - 5.0/6.0);
    
    double delta_lambda = (term_t + term_W + term_Z) / (16.0 * PI2);
    return p.lambda + delta_lambda;
}

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
    double term2 = g2_4/LOOP2 * (
        12*g3_2 + 35*g2_2/6 + 9*g1_2/10 - 3*yt2/2 - 3*yb2/2 - ytau2/2
    );
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
    double term2 = g3_4/LOOP2 * (
        -26*g3_2 + 9*g2_2/2 + 11*g1_2/10 - 2*yt2 - 2*yb2
    );
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
        3*yt4 - 3*yb4 - ytau4 +
        9*g2_4/16 + 27*g1_4/400 + 9*g2_2*g1_2/40
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


static constexpr double CLAUDE_M_PLANCK = 1.22e19;
static constexpr double ALPHA      = 4.0;
static constexpr double SIGMA_PLAN = 8.0;
static constexpr int    N_QUAD     = 2048;

static double get_pure_sm_lambda(RGEHelper& rge, double phi_dimless, double mu_inst) {
    double h = phi_dimless * mu_inst;
    double h_rge = std::min(h, CLAUDE_M_PLANCK);
    if (h_rge <= 246.22) return 0.0;

    double t      = 2.0 * std::log(h_rge);
    Params p      = rge.get_params(t);
    return get_lambda_eff(p);
}

static double integrand_u(RGEHelper& rge, double u, double R, double mu_inst, double prefactor) {
    if (std::abs(u) >= 1.0) return 0.0;
    double ratio = (1.0 + u) / (1.0 - u);
    double e2x   = std::pow(ratio, ALPHA);
    double e4x   = e2x * e2x;
    double phi_x  = prefactor * 2.0 / (R * (e2x + 1.0));
    
    double lam_phi = get_pure_sm_lambda(rge, phi_x, mu_inst);
    double phi4 = std::pow(phi_x, 4);
    
    double V_SM = 0.25 * lam_phi * phi4;
    
    double lambda_6 = 1.0;
    double V_6 = (lambda_6 / 6.0) * std::pow(phi_x * mu_inst / CLAUDE_M_PLANCK, 2) * phi4;
    
    double V_x = V_SM + V_6;
    
    double jacobian = ALPHA / (1.0 - u * u);
    // Use pi*pi instead of LOOP1 (which is 16*pi^2)
    return 2.0 * pi * pi * std::pow(R, 4) * e4x * V_x * jacobian;
}

double evaluate_action_at_R(RGEHelper& rge, double mu_inst, double R) {
    if (R <= 0.0) return 1e100;
    double mu_R = mu_inst / R;
    if (mu_R <= 0.0) return 1e100;
    double t_R    = 2.0 * std::log(mu_R);
    Params p_R    = rge.get_params(t_R);
    double lambda_R = p_R.lambda;
    if (lambda_R >= 0.0) return 1e100;
    double abs_lam  = std::abs(lambda_R);
    double prefactor = std::sqrt(2.0 / abs_lam);
    // The kinetic integral for the conformal bounce is exactly 16 pi^2 / (3 |\lambda|)
    double kinetic_term = (16.0 * pi * pi) / (3.0 * abs_lam);
    if (kinetic_term > 5e5) return kinetic_term;
    const int    N  = N_QUAD;
    const double du = 2.0 / N;
    double potential_integral = 0.0;
    for (int i = 0; i <= N; ++i) {
        double u = -1.0 + i * du;
        if (i == 0 || i == N) continue;
        double f = integrand_u(rge, u, R, mu_inst, prefactor);
        double weight;
        if (i % 2 == 1) weight = 4.0 / 3.0;
        else            weight = 2.0 / 3.0;
        potential_integral += weight * f * du;
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
    double tol = 1e-13;
    
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
std::tuple<int, double, double, double> classify_stability(double Mh, double Mt) {
    Params y = get_nnlo_matching(Mh, Mt);
    double Mh_calc = Mh; // Precise mass matching is embedded inside get_nnlo_matching
    
    double t0 = 2*log(Mt);
    double tPlanck = 2*log(MPlanck);
    
    RGEHelper rge;
    bool is_unstable = false;
    double mu_inst = MPlanck;
    
    double min_lambda_eff = 1.0;
    double t_min_lambda = t0;
    
    double t = t0;
    double dt = 0.1;
    rge.add_point(t, y);
    
    double t_v = 2 * log(v);
    
    if (t >= t_v) {
        min_lambda_eff = get_lambda_eff(y);
        if (min_lambda_eff <= 0.0) {
            mu_inst = exp(t/2.0);
            is_unstable = true;
        }
    }
    
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        
        y = rk4_single_step(y, t, dt);
        t += dt;
        rge.add_point(t, y);
        
        if (std::abs(y.lambda) > 4*pi || y.yt > 4*pi) 
            return std::make_tuple(4, -1.0, -1.0, Mh_calc);
            
        if (!std::isfinite(y.lambda) || !std::isfinite(y.yt))
            return std::make_tuple(4, -1.0, -1.0, Mh_calc);
            
        if (t >= t_v) {
            double lam_eff = get_lambda_eff(y);
            if (lam_eff < min_lambda_eff) {
                min_lambda_eff = lam_eff;
                t_min_lambda = t;
            }
            if (!is_unstable && lam_eff < 0) {
                mu_inst = exp(t/2.0);
                is_unstable = true;
            }
        }
    }
    
    if (!is_unstable) return std::make_tuple(1, -1.0, -1.0, Mh_calc); // Stable
    
    double S_approx = 8.0 * pi * pi / (3.0 * std::abs(min_lambda_eff));
    double S_exact = find_minimum_action(rge, mu_inst, t_min_lambda);
    
    // Exact Metastability Threshold using Universe Age
    double Tv = 1.179e44 / v;
    double S_threshold = 4.0 * std::log(Tv * mu_inst);
    
    if (S_exact > S_threshold) return std::make_tuple(2, S_exact, S_approx, Mh_calc); // Metastable
    return std::make_tuple(3, S_exact, S_approx, Mh_calc); // Unstable
}


// --- STRESS TEST ODE SOLVER ---
#include <iomanip>

double V_eff(RGEHelper& rge, double phi) {
    if (phi <= v) return 0.0;
    double t = 2.0 * std::log(phi);
    Params p = rge.get_params(t);
    double lam = get_lambda_eff(p);
    double V_SM = 0.25 * lam * std::pow(phi, 4);
    double V_6 = (1.0 / 6.0) * std::pow(phi, 6) / (CLAUDE_M_PLANCK * CLAUDE_M_PLANCK);
    return V_SM + V_6;
}

double dV_eff(RGEHelper& rge, double phi) {
    if (phi <= v) return 0.0;
    double h = phi * 1e-5;
    double v_m2 = V_eff(rge, phi - 2*h);
    double v_m1 = V_eff(rge, phi - h);
    double v_p1 = V_eff(rge, phi + h);
    double v_p2 = V_eff(rge, phi + 2*h);
    return (v_m2 - 8*v_m1 + 8*v_p1 - v_p2) / (12.0 * h);
}

// 1. Direct Evaluation (No Interpolation)
struct DirectEvaluator {
    RGEHelper* rge;
    void eval(long double phi, long double &V, long double &dV) {
        V = V_eff(*rge, (double)phi);
        dV = dV_eff(*rge, (double)phi);
    }
};

// 2. Linear Interpolator (Baseline)
struct LinearInterp {
    long double min_log_phi, max_log_phi, d_log;
    std::vector<long double> V_vals;
    std::vector<long double> dV_vals;
    
    void init(RGEHelper& rge, double phi_min, double phi_max, int N) {
        min_log_phi = std::log(phi_min);
        max_log_phi = std::log(phi_max);
        d_log = (max_log_phi - min_log_phi) / (N - 1);
        V_vals.resize(N);
        dV_vals.resize(N);
        for(int i=0; i<N; ++i) {
            double phi = std::exp(min_log_phi + i * d_log);
            V_vals[i] = V_eff(rge, phi);
            dV_vals[i] = dV_eff(rge, phi);
        }
    }
    
    void eval(long double phi, long double &V, long double &dV) {
        if (phi <= v) { V = 0.0; dV = 0.0; return; }
        if (phi <= std::exp(min_log_phi)) {
            long double lam0 = V_vals[0] * 4.0 / std::pow(std::exp(min_log_phi), 4);
            V = 0.25 * lam0 * std::pow(phi, 4);
            dV = lam0 * std::pow(phi, 3);
            return;
        }
        if (phi >= std::exp(max_log_phi)) {
            V = V_vals.back(); dV = dV_vals.back(); return;
        }
        long double lphi = std::log(phi);
        long double idx_d = (lphi - min_log_phi) / d_log;
        int idx = (int)idx_d;
        long double f = idx_d - idx;
        V = V_vals[idx] + f * (V_vals[idx+1] - V_vals[idx]);
        dV = dV_vals[idx] + f * (dV_vals[idx+1] - dV_vals[idx]);
    }
};

struct State {
    long double phi;
    long double dphi;
};

template<typename Evaluator>
State ode_step(Evaluator& pot, State y, long double r, long double dr) {
    auto derivs = [&](long double curr_r, State curr_y) -> State {
        long double V, dV;
        pot.eval(curr_y.phi, V, dV);
        long double friction = (curr_r == 0) ? 0 : (3.0 / curr_r) * curr_y.dphi;
        return { curr_y.dphi, -friction + dV };
    };
    
    State k1 = derivs(r, y);
    State k2 = derivs(r + 0.5*dr, {y.phi + 0.5*dr*k1.phi, y.dphi + 0.5*dr*k1.dphi});
    State k3 = derivs(r + 0.5*dr, {y.phi + 0.5*dr*k2.phi, y.dphi + 0.5*dr*k2.dphi});
    State k4 = derivs(r + dr, {y.phi + dr*k3.phi, y.dphi + dr*k3.dphi});
    
    return {
        y.phi + (dr/6.0)*(k1.phi + 2*k2.phi + 2*k3.phi + k4.phi),
        y.dphi + (dr/6.0)*(k1.dphi + 2*k2.dphi + 2*k3.dphi + k4.dphi)
    };
}

template<typename Evaluator>
long double shoot(Evaluator& pot, long double phi_0, long double& S_kin, long double& S_pot, 
                 long double tol_dr=0.01, long double start_r=1e-4) {
    long double r = start_r;
    long double V0, dV0;
    pot.eval(phi_0, V0, dV0);
    
    State y;
    y.phi = phi_0 + 0.125 * r * r * dV0;
    y.dphi = 0.25 * r * dV0;
    
    long double dr = start_r;
    long double max_r = 1e8;
    
    S_kin = 0;
    S_pot = 0;
    long double PI = 3.14159265358979323846L;
    
    while (r < max_r) {
        long double V, dV;
        pot.eval(y.phi, V, dV);
        
        State y_next = ode_step(pot, y, r, dr);
        
        long double r3 = r * r * r;
        long double kin_term = 0.5 * y.dphi * y.dphi;
        long double r_next = r + dr;
        long double r3_next = r_next * r_next * r_next;
        
        long double V_next, dV_next;
        pot.eval(y_next.phi, V_next, dV_next);
        long double kin_term_next = 0.5 * y_next.dphi * y_next.dphi;
        
        S_kin += 2.0 * PI * PI * 0.5 * (r3 * kin_term + r3_next * kin_term_next) * dr;
        S_pot += 2.0 * PI * PI * 0.5 * (r3 * V + r3_next * V_next) * dr;
        
        y = y_next;
        r += dr;
        
        if (y.phi < v && y.dphi < 0) return y.phi; 
        if (y.dphi > 0) return y.phi; 
        if (y.phi < 0) return y.phi; 
        
        long double dr_target = tol_dr * std::abs(y.phi / (y.dphi + 1e-30L));
        dr = std::max(1e-6L, std::min(dr_target, 1.0L));
    }
    return y.phi;
}


template<typename Evaluator>
double find_root(Evaluator& pot, long double phi_low, long double phi_high, 
                 long double tol_dr, long double start_r, double& out_Skin, double& out_Spot) {
                 
    // Bracket search
    bool found_overshoot = false;
    long double dummy1, dummy2;
    while(phi_high < 1e18) {
        long double res = shoot(pot, phi_high, dummy1, dummy2, tol_dr, start_r);
        if (res < 0) { found_overshoot = true; break; }
        phi_low = phi_high;
        phi_high *= 10.0;
    }
    
    if (!found_overshoot) return -1.0;
                 
    long double phi_opt = 0;
    long double S_kin = 0, S_pot = 0;
    
    for (int iter = 0; iter < 100; ++iter) {
        phi_opt = std::exp(0.5 * (std::log(phi_low) + std::log(phi_high)));
        long double end_val = shoot(pot, phi_opt, S_kin, S_pot, tol_dr, start_r);
        
        if (end_val < 0) phi_high = phi_opt;
        else phi_low = phi_opt;
        
        if (std::abs(phi_high - phi_low) / phi_opt < 1e-14) {
            break;
        }
    }
    out_Skin = (double)S_kin;
    out_Spot = (double)S_pot;
    return (double)(S_kin + S_pot);
}


int main() {
    double Mh = 5.0;
    double Mt = 105.0;
    
    std::cout << "Initializing RGEs for stress testing at Mh = 5.0, Mt = 105.0" << std::endl;
    
    Params y = get_nnlo_matching(Mh, Mt);
    double t = 2 * std::log(Mt);
    double tPlanck = 2 * std::log(MPlanck);
    double dt = 0.1;
    
    RGEHelper rge;
    rge.add_point(t, y);
    
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        y = rk4_single_step(y, t, dt);
        t += dt;
        rge.add_point(t, y);
    }
    
    LinearInterp pot_interp;
    pot_interp.init(rge, 100.0, 1e18, 1000000); 
    
    DirectEvaluator pot_direct;
    pot_direct.rge = &rge;
    
    std::ofstream fout("data/stress_test_results.csv");
    fout << "Test_Name,S_total,S_kin,S_pot\n" << std::flush;
    
    double s_kin, s_pot, s_tot;
    
    // Baseline
    std::cout << "Running Baseline..." << std::endl;
    s_tot = find_root(pot_interp, 100.0, 1e8, 0.01, 1e-4, s_kin, s_pot);
    fout << "Baseline," << s_tot << "," << s_kin << "," << s_pot << "\n" << std::flush;
    
    // 1. Different phi0 brackets
    std::cout << "Test 1: Wide bracket..." << std::endl;
    s_tot = find_root(pot_interp, 10.0, 1e14, 0.01, 1e-4, s_kin, s_pot);
    fout << "Wide Bracket (10 to 1e14)," << s_tot << "," << s_kin << "," << s_pot << "\n" << std::flush;
    
    // 2. Different RK4 tolerances
    std::cout << "Test 2: RK4 tol = 1e-3..." << std::endl;
    s_tot = find_root(pot_interp, 100.0, 1e8, 0.001, 1e-4, s_kin, s_pot);
    fout << "RK4 tol 1e-3," << s_tot << "," << s_kin << "," << s_pot << "\n" << std::flush;
    
    std::cout << "Test 2: RK4 tol = 1e-4..." << std::endl;
    s_tot = find_root(pot_interp, 100.0, 1e8, 0.0001, 1e-4, s_kin, s_pot);
    fout << "RK4 tol 1e-4," << s_tot << "," << s_kin << "," << s_pot << "\n" << std::flush;
    
    // 3. Different starting radii
    std::cout << "Test 3: Start radius = 1e-6..." << std::endl;
    s_tot = find_root(pot_interp, 100.0, 1e8, 0.01, 1e-6, s_kin, s_pot);
    fout << "Start r=1e-6," << s_tot << "," << s_kin << "," << s_pot << "\n" << std::flush;
    
    std::cout << "Test 3: Start radius = 1e-2..." << std::endl;
    s_tot = find_root(pot_interp, 100.0, 1e8, 0.01, 1e-2, s_kin, s_pot);
    fout << "Start r=1e-2," << s_tot << "," << s_kin << "," << s_pot << "\n" << std::flush;
    
    
    
    fout.close();
    std::cout << "Stress tests complete." << std::endl;
    return 0;
}
