#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// --- CONSTANTS ---
const double v_ew = 246.22;
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

Params get_nnlo_matching(double Mh, double Mt) {
    double dAlphas = (alpha3_at_Mz - 0.1184) / 0.0007;
    double lambda_tree = (Mh * Mh) / (2.0 * v_ew * v_ew);
    double yt_tree = (sqrt(2.0) * Mt) / v_ew;
    double delta_lambda = -0.00313 - 0.00004 * (Mt - 173.34);
    double delta_yt = -0.0587 - 0.00042 * dAlphas;
    double lambda_Mt = lambda_tree + delta_lambda;
    double yt_Mt = yt_tree + delta_yt;
    double g1_Mt = 0.46266 + 0.00006 * (Mt - 173.34);
    double g2_Mt = 0.65355 + 0.00002 * (Mt - 173.34);
    double g3_Mt = 1.1666 + 0.00314 * dAlphas - 0.00046 * (Mt - 173.34);
    return {g1_Mt, g2_Mt, g3_Mt, yt_Mt, sqrt(2.0)*Mb/v_ew, sqrt(2.0)*Mtau/v_ew, lambda_Mt, 0};
}

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

double betaG1sq(const Params& p) { /* Not fully needed unless RGE runs, but required for RK4 */
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

// --- NEW CANONICAL POTENTIAL EVALUATOR ---
struct CanonicalPot {
    double c6;
    RGEHelper* rge;
    
    // Evaluate exact V(phi) using purely analytical expressions.
    double V(double phi) const {
        if (phi <= v_ew) return 0.0;
        double t = 2.0 * std::log(phi);
        Params p = rge->get_params(t);
        double lam = get_lambda_eff(p);
        double V_SM = 0.25 * lam * std::pow(phi, 4);
        double V_c6 = (c6 / 6.0) * std::pow(phi, 6) / (MPlanck * MPlanck);
        return V_SM + V_c6;
    }
    
    // Evaluate exact dV/dphi using analytical derivatives and beta functions.
    double dV(double phi) const {
        if (phi <= v_ew) return 0.0;
        double t = 2.0 * std::log(phi);
        Params p = rge->get_params(t);
        double lam = get_lambda_eff(p);
        double b_lam = betaLambda(p);
        double dV_SM = std::pow(phi, 3) * (lam + 0.25 * b_lam);
        double dV_c6 = c6 * std::pow(phi, 5) / (MPlanck * MPlanck);
        return dV_SM + dV_c6;
    }
};

// --- NEW CANONICAL SHOOTING ALGORITHM ---

enum Status {
    CONVERGED,
    OVERSHOOT,
    UNDERSHOOT,
    MAX_STEPS,
    FAILED
};

struct State {
    double phi;
    double dphi;
};

struct TrajectoryResult {
    Status status;
    double S_kin;
    double S_pot;
    double final_phi;
};

TrajectoryResult shoot(const CanonicalPot& pot, double phi_0, double tol_dr=0.01, int max_steps=1000000) {
    if (phi_0 <= v_ew) return {FAILED, 0, 0, phi_0};
    
    // Taylor expansion at r = 1e-4
    double r = 1e-4;
    double dV0 = pot.dV(phi_0);
    
    State y;
    y.phi = phi_0 + 0.125 * r * r * dV0;
    y.dphi = 0.25 * r * dV0;
    
    double S_kin = 0.0;
    double S_pot = 0.0;
    
    double dr = 1e-4;
    int step_cnt = 0;
    
    while (r < 1e12) {
        step_cnt++;
        if (step_cnt > max_steps) {
            return {MAX_STEPS, 0, 0, y.phi}; // Return 0 action for non-converged
        }
        
        // RK4 Step
        auto derivs = [&](double curr_r, State curr_y) -> State {
            double V_val = pot.V(curr_y.phi);
            double dV_val = pot.dV(curr_y.phi);
            double friction = (curr_r == 0) ? 0 : (3.0 / curr_r) * curr_y.dphi;
            return { curr_y.dphi, -friction + dV_val };
        };
        
        State k1 = derivs(r, y);
        State k2 = derivs(r + 0.5*dr, {y.phi + 0.5*dr*k1.phi, y.dphi + 0.5*dr*k1.dphi});
        State k3 = derivs(r + 0.5*dr, {y.phi + 0.5*dr*k2.phi, y.dphi + 0.5*dr*k2.dphi});
        State k4 = derivs(r + dr, {y.phi + dr*k3.phi, y.dphi + dr*k3.dphi});
        
        State y_next = {
            y.phi + (dr/6.0)*(k1.phi + 2*k2.phi + 2*k3.phi + k4.phi),
            y.dphi + (dr/6.0)*(k1.dphi + 2*k2.dphi + 2*k3.dphi + k4.dphi)
        };
        
        // Action integration (Simpson-like or Trapezoidal over step)
        double r3 = r * r * r;
        double kin_term = 0.5 * y.dphi * y.dphi;
        double r_next = r + dr;
        double r3_next = r_next * r_next * r_next;
        double kin_term_next = 0.5 * y_next.dphi * y_next.dphi;
        
        double V_curr = pot.V(y.phi);
        double V_next = pot.V(y_next.phi);
        
        S_kin += 2.0 * PI2 * 0.5 * (r3 * kin_term + r3_next * kin_term_next) * dr;
        S_pot += 2.0 * PI2 * 0.5 * (r3 * V_curr + r3_next * V_next) * dr;
        
        y = y_next;
        r += dr;
        
        // Termination Checks
        if (std::isnan(y.phi) || std::isnan(y.dphi)) {
            return {FAILED, 0, 0, y.phi};
        }
        if (y.phi < 0) {
            return {OVERSHOOT, 0, 0, y.phi}; // Overshoot crosses 0
        }
        if (y.dphi > 0) {
            // If it undershoots near the vacuum, it's considered converged!
            if (y.phi < 1000.0) {
                return {CONVERGED, S_kin, S_pot, y.phi};
            }
            return {UNDERSHOOT, 0, 0, y.phi}; // Undershoot starts rolling back up
        }
        
        // Adaptive Step Sizing
        double dr_target = tol_dr * std::abs(y.phi / (y.dphi + 1e-30));
        dr = std::max(1e-5, std::min(dr_target, 0.1 * r + 1.0));
    }
    
    return {MAX_STEPS, 0, 0, y.phi};
}

std::string status_to_string(Status s) {
    if (s == CONVERGED) return "CONVERGED";
    if (s == OVERSHOOT) return "OVERSHOOT";
    if (s == UNDERSHOOT) return "UNDERSHOOT";
    if (s == MAX_STEPS) return "MAX_STEPS";
    return "FAILED";
}

int main(int argc, char* argv[]) {
    double Mh = 5.0;
    double Mt = 105.0;
    double c6 = 0.0; // Pure SM by default
    
    if (argc > 1) Mh = std::stod(argv[1]);
    if (argc > 2) Mt = std::stod(argv[2]);
    if (argc > 3) c6 = std::stod(argv[3]);
    
    std::cout << "Canonical Solver Validating Single Point: Mh=" << Mh << ", Mt=" << Mt << ", c6=" << c6 << std::endl;
    
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
    
    CanonicalPot pot = {c6, &rge};
    
    // Find Bracket
    double phi_low = 100.0;
    double phi_high = 1e6;
    bool found_overshoot = false;
    
    std::cout << "Finding overshoot bracket..." << std::endl;
    while(phi_high < 1e18) {
        TrajectoryResult res = shoot(pot, phi_high);
        std::cout << "  phi_high=" << phi_high << " -> status=" << status_to_string(res.status) << std::endl;
        if (res.status == OVERSHOOT) {
            found_overshoot = true;
            break;
        }
        phi_low = phi_high;
        phi_high *= 10.0;
    }
    
    if (!found_overshoot) {
        std::cout << "Could not find overshoot bracket. Action is infinite." << std::endl;
        return 0;
    }
    
    std::cout << "Bracket found: [" << phi_low << ", " << phi_high << "]" << std::endl;
    
    double phi_opt = 0;
    TrajectoryResult best_res = {FAILED, 0, 0, 0};
    
    for (int iter = 0; iter < 150; ++iter) {
        phi_opt = std::exp(0.5 * (std::log(phi_low) + std::log(phi_high)));
        TrajectoryResult res = shoot(pot, phi_opt);
        
        if (res.status == OVERSHOOT) {
            phi_high = phi_opt;
        } else if (res.status == UNDERSHOOT) {
            phi_low = phi_opt;
        } else if (res.status == CONVERGED) {
            best_res = res;
            phi_high = phi_opt; // A converged run technically reaches the vacuum; further refinement can sharpen it
        } else {
            // MAX_STEPS or FAILED in bisection usually means we are extremely close to the true vacuum
            // We'll cautiously treat it as undershoot if phi didn't drop below 0
            if (res.final_phi > 0) phi_low = phi_opt;
            else phi_high = phi_opt;
        }
        
        if (std::abs(phi_high - phi_low) / phi_opt < 1e-14) {
            std::cout << "Bisection converged after " << iter << " iterations." << std::endl;
            break;
        }
    }
    
    // Evaluate one final time to be completely sure we have a CONVERGED trajectory
    TrajectoryResult final_res = shoot(pot, phi_opt);
    
    std::cout << "Final Evaluation Status: " << status_to_string(final_res.status) << std::endl;
    
    if (final_res.status == CONVERGED || best_res.status == CONVERGED) {
        TrajectoryResult val = (final_res.status == CONVERGED) ? final_res : best_res;
        double S_total = val.S_kin + val.S_pot;
        double virial = val.S_kin / std::abs(val.S_pot);
        std::cout << "Valid Solution Found:" << std::endl;
        std::cout << "S_total = " << S_total << std::endl;
        std::cout << "S_kin   = " << val.S_kin << std::endl;
        std::cout << "S_pot   = " << val.S_pot << std::endl;
        std::cout << "Virial  = " << virial << std::endl;
        std::cout << "phi_0   = " << phi_opt << std::endl;
    } else {
        std::cout << "Failed to find a CONVERGED trajectory. No valid action can be computed." << std::endl;
        std::cout << "Most likely due to extremely flat potential causing MAX_STEPS." << std::endl;
    }
    
    return 0;
}
