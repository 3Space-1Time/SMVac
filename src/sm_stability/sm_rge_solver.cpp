#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <tuple>

using namespace std;

// --- PHYSICAL CONSTANTS ---
const double v = 246.22, MPlanck = 1.22e19, Mtau = 1.777, Mb = 4.0;
const double g1init = 0.46, g2init = 0.65, g3init = 1.1666;
const double pi = M_PI, pi2 = pow(4*pi, 2), pi4 = pi2*pi2, pi6 = pi2*pi2*pi2, pi8 = pi4*pi4;
const double T_universe_GeV_inv = 4.79e41;
const double alpha3_at_Mz = 0.1184; 

struct Params {
    double g1, g2, g3, yt, yb, ytau, lambda;
    double phi = 0; // Parameter for mass calculation
};

// --- RENORMALIZATION GROUP EQUATIONS (RGEs) ---
#pragma region Beta Functions
double betaG1sq(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2, g3_2 = p.g3*p.g3;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2, g3_4 = g3_2*g3_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    
    double term1 = g1_4/pi2 * (41.0/10);
    double term2 = g1_4/pi4 * (44*g3_2/5 + 27*g2_2/10 + 199*g1_2/50 - 17*yt2/10 - yb2/2 - 3*ytau2/2);
    double term3 = g1_4/pi6 * (
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
    
    double term1 = g2_4/pi2 * (-19.0/6);
    double term2 = g2_4/pi4 * (
        12*g3_2 + 35*g2_2/6 + 9*g1_2/10 - 3*yt2/2 - 3*yb2/2 - ytau2/2
    );
    double term3 = g2_4/pi6 * (
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
    
    double term1 = -g3_4/pi2 * 7;
    double term2 = g3_4/pi4 * (
        -26*g3_2 + 9*g2_2/2 + 11*g1_2/10 - 2*yt2 - 2*yb2
    );
    double term3 = g3_4/pi6 * (
        yt2*(15*yt2 - 40*g3_2 - 93*g2_2/8 - 101*g1_2/40) +
        65*g3_4/2 + 109*g2_4/8 - 523*g1_4/120 +
        21*g3_2*g2_2 + 77*g3_2*g1_2/15 - 3*g2_2*g1_2/40
    );
    double term4 = g3_10/pi8 * 2472.28;
    
    return term1 + term2 + term3 + term4;
}
double betaLambda(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2, g3_2 = p.g3*p.g3;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2, g3_4 = g3_2*g3_2;
    double g1_6 = g1_4*g1_2, g2_6 = g2_4*g2_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    double yt4 = yt2*yt2, yb4 = yb2*yb2, ytau4 = ytau2*ytau2;
    
    double term1 = (1/pi2) * (
        p.lambda*(12*p.lambda + 6*yt2 + 6*yb2 + 2*ytau2 - 9*g2_2/2 - 9*g1_2/10) -
        3*yt4 - 3*yb4 - ytau4 +
        9*g2_4/16 + 27*g1_4/400 + 9*g2_2*g1_2/40
    );
    
    double term2 = (1/pi4) * (
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
    
    double term3 = (1/pi6) * (
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
    
    double term1 = yt2/pi2 * ( 9*yt2/2 + 3*yb2/2 + ytau2 - 8*g3_2 - 9*g2_2/4 - 17*g1_2/20 );
    double term2 = yt2/pi4 * ( yt2*(-12*yt2 - 11*yb2/4 - 9*ytau2/4 - 12*p.lambda + 36*g3_2 + 225*g2_2/16 + 393*g1_2/80) + yb2*(-yb2/4 + 5*ytau2/4 + 4*g3_2 + 99*g2_2/16 + 7*g1_2/80) + ytau2*(-9*ytau2/4 + 15*g2_2/8 + 15*g1_2/8) + 6*p.lambda*p.lambda - 108*g3_4 - 23*g2_4/4 + 1187*g1_4/600 + 9*g3_2*g2_2 + 19*g3_2*g1_2/15 - 9*g2_2*g1_2/20 );
    double term3 = yt2/pi6 * ( yt2*(58.6028*yt2 + 198*p.lambda - 157*g3_2 - 1593*g2_2/16 - 2437*g1_2/80) + p.lambda*yt2*(15*p.lambda/4 + 16*g3_2 - 135*g2_2/2 - 127*g1_2/10) + yt2*(363.764*g3_4 + 16.990*g2_4 - 24.422*g1_4 + 48.370*g3_2*g2_2 + 18.074*g3_2*g1_2 + 34.829*g2_2*g1_2) + p.lambda*p.lambda*(-36*p.lambda + 45*g2_2 + 9*g1_2) + p.lambda*(-171*g2_4/16 - 1089*g1_4/400 + 117*g2_2*g1_2/40) - 619.35*g3_6 + 169.829*g2_4*g2_2 + 16.099*g1_4*g1_2 + 73.654*g3_4*g2_2 - 15.096*g3_4*g1_2 - 21.072*g3_2*g2_4 - 22.319*g3_2*g1_4 - 321*g3_2*g2_2*g1_2/20 - 4.743*g2_4*g1_2 - 4.442*g2_2*g1_4 );
    return term1 + term2 + term3;
}
double betaYb2(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2, g3_2 = p.g3*p.g3;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2, g3_4 = g3_2*g3_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    
    double term1 = yb2/pi2 * ( 3*yt2/2 + 9*yb2/2 + ytau2 - 8*g3_2 - 9*g2_2/4 - g1_2/4 );
    double term2 = yb2/pi4 * ( yt2*(-yt2/4 - 11*yb2/4 + 5*ytau2/4 + 4*g3_2 + 99*g2_2/16 + 91*g1_2/80) + yb2*(-12*yb2 - 9*ytau2/4 - 12*p.lambda + 36*g3_2 + 225*g2_2/16 + 237*g1_2/80) + ytau2*(-9*ytau2/4 + 15*g2_2/8 + 15*g1_2/8) + 6*p.lambda*p.lambda - 108*g3_4 - 23*g2_4/4 - 127*g1_4/600 + 9*g3_2*g2_2 + 31*g3_2*g1_2/15 - 27*g2_2*g1_2/20 );
    return term1 + term2;
}
double betaYtau2(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2;
    double g1_4 = g1_2*g1_2, g2_4 = g2_2*g2_2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    
    double term1 = ytau2/pi2 * ( 3*yt2 + 3*yb2 + 5*ytau2/2 - 9*g2_2/4 - 9*g1_2/4 );
    double term2 = ytau2/pi4 * ( 6*p.lambda*p.lambda - 23*g2_4/4 + 1371*g1_4/200 + 27*g2_2*g1_2/20 + yt2*(-27*yt2/4 + 3*yb2/2 - 27*ytau2/4 + 20*p.g3*p.g3 + 45*g2_2/8 + 17*g1_2/8) + yb2*(-27*yb2/4 - 27*ytau2/4 + 20*p.g3*p.g3 + 45*g2_2/8 + 5*g1_2/8) + ytau2*(-3*ytau2 - 12*p.lambda + 165*g2_2/16 + 537*g1_2/80) );
    return term1 + term2;
}
#pragma endregion

// Beta function for the phi parameter, needed for mass calculation
double beta_phi_1loop(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    return (1/pi2) * (6*p.lambda + 3*yt2 + 3*yb2 + ytau2 - 9*g1_2/20 - 9*g2_2/4);
}
// --- RK4 SOLVER ---
void rk4_step(Params& y, double t, double dt, bool with_phi = false) {
    auto dydt = [&](const Params& p) -> Params {
        Params dy;
        dy.g1 = 0.5/p.g1 * betaG1sq(p);
        dy.g2 = 0.5/p.g2 * betaG2sq(p);
        dy.g3 = 0.5/p.g3 * betaG3sq(p);
        dy.yt = 0.5/p.yt * betaYt2(p);
        dy.yb = 0.5/p.yb * betaYb2(p);
        dy.ytau = 0.5/p.ytau * betaYtau2(p);
        dy.lambda = betaLambda(p);
        dy.phi = with_phi ? beta_phi_1loop(p) : 0;
        return dy;
    };
    
    Params k1 = dydt(y);
    Params k2 = dydt({y.g1 + 0.5*dt*k1.g1, y.g2 + 0.5*dt*k1.g2, y.g3 + 0.5*dt*k1.g3,
                      y.yt + 0.5*dt*k1.yt, y.yb + 0.5*dt*k1.yb, y.ytau + 0.5*dt*k1.ytau,
                      y.lambda + 0.5*dt*k1.lambda, y.phi + 0.5*dt*k1.phi});
    Params k3 = dydt({y.g1 + 0.5*dt*k2.g1, y.g2 + 0.5*dt*k2.g2, y.g3 + 0.5*dt*k2.g3,
                      y.yt + 0.5*dt*k2.yt, y.yb + 0.5*dt*k2.yb, y.ytau + 0.5*dt*k2.ytau,
                      y.lambda + 0.5*dt*k2.lambda, y.phi + 0.5*dt*k2.phi});
    Params k4 = dydt({y.g1 + dt*k3.g1, y.g2 + dt*k3.g2, y.g3 + dt*k3.g3,
                      y.yt + dt*k3.yt, y.yb + dt*k3.yb, y.ytau + dt*k3.ytau,
                      y.lambda + dt*k3.lambda, y.phi + dt*k3.phi});
    
    y.g1 += dt/6 * (k1.g1 + 2*k2.g1 + 2*k3.g1 + k4.g1);
    y.g2 += dt/6 * (k1.g2 + 2*k2.g2 + 2*k3.g2 + k4.g2);
    y.g3 += dt/6 * (k1.g3 + 2*k2.g3 + 2*k3.g3 + k4.g3);
    y.yt += dt/6 * (k1.yt + 2*k2.yt + 2*k3.yt + k4.yt);
    y.yb += dt/6 * (k1.yb + 2*k2.yb + 2*k3.yb + k4.yb);
    y.ytau += dt/6 * (k1.ytau + 2*k2.ytau + 2*k3.ytau + k4.ytau);
    y.lambda += dt/6 * (k1.lambda + 2*k2.lambda + 2*k3.lambda + k4.lambda);
    if (with_phi) y.phi += dt/6 * (k1.phi + 2*k2.phi + 2*k3.phi + k4.phi);
}

// *** CLASSIFICATION LOGIC ***
int classify_stability(double Mh, double Mt) {
    double lambda0 = Mh*Mh / (2*v*v);
    double yb0 = sqrt(2)*Mb/v, ytau0 = sqrt(2)*Mtau/v;
    
    // --- KEY FIX: Use NNLO Matching for yt0 from main.cpp ---
    // Old: double yt0 = sqrt(2)*Mt/v;
    // New:
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    // --------------------------------------------------------

    double t0 = 2*log(172.5); // Start at Top mass scale approx
    double tPlanck = 2*log(MPlanck);
    
    Params y = {g1init, g2init, g3init, yt0, yb0, ytau0, lambda0, 0};
    
    int n_steps = 4000;
    double dt = (tPlanck - t0) / n_steps;
    double lambda_min = lambda0;
    double mu_inst = MPlanck;
    bool is_unstable = false;

    for (int i = 0; i < n_steps; ++i) {
        double t = t0 + i * dt;
        rk4_step(y, t, dt, false);
        
        // Non-perturbative check (Lambda > 4*pi or couplings blow up)
        if (y.lambda > 4.0*pi || y.g1 > 10.0 || y.yt > 10.0) return 4; 
        if (!isfinite(y.lambda) || !isfinite(y.yt)) return 4;

        if (y.lambda < lambda_min) lambda_min = y.lambda;
        
        // Check for instability crossing
        if (!is_unstable && y.lambda <= 0) {
            mu_inst = exp(t/2); // mu = e^(t/2)
            is_unstable = true;
        }
    }
    
    if (lambda_min > 0) return 1; // Stable
    
    // --- Metastability Check using Analytic Approximation ---
    if (is_unstable) {
        double R = 1.0/mu_inst;
        // Analytic threshold formula requested
        double threshold = -0.065 / (1 - 0.01*log(R*v));
        return (lambda_min > threshold) ? 2 : 3; // 2=Metastable, 3=Unstable
    }
    
    return 3;
}

double compute_higgs_mass(double Mt, double Mh_input) {
    double lambda0 = Mh_input*Mh_input / (2*v*v);
    double yb0 = sqrt(2)*Mb/v, ytau0 = sqrt(2)*Mtau/v;
    double t0 = 2*log(172.5), tv = 2*log(v);
    
    // --- KEY FIX: Consistent NNLO Matching here too ---
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    // --------------------------------------------------
    
    Params y = {g1init, g2init, g3init, yt0, yb0, ytau0, lambda0, 0};
    
    double dt = (tv - t0) / 4000.0;
    for (int i = 0; i < 4000; ++i) {
        rk4_step(y, t0 + i*dt, dt, true);
        if (!isfinite(y.g1) || !isfinite(y.lambda)) return -1;
    }
    
    double V_prime_prime = 2.0 * y.lambda * v*v; 
    if (V_prime_prime < 0) return -1.0;
    
    return sqrt(V_prime_prime);
}

int classify_point(double Mt, double Mh_input, double& calculated_mh) {
    calculated_mh = compute_higgs_mass(Mt, Mh_input);
    if (calculated_mh < 0) return 4; // Invalid
    
    // Use the calculated mass for the stability check? 
    // Typically stability plots are in terms of Pole Masses (Input Mt, Calculated Mh).
    // The classify_stability function takes (Mh, Mt).
    return classify_stability(calculated_mh, Mt); 
}

void process_chunk(const vector<pair<double,double>>& points, 
                   vector<tuple<double,double,double,int>>& results, 
                   int start, int end) {
    for (int i = start; i < end; ++i) {
        double Mt = points[i].first;
        double Mh_input = points[i].second;
        double calculated_mh;
        int stability = classify_point(Mt, Mh_input, calculated_mh);
        results[i] = make_tuple(Mt, calculated_mh, Mh_input, stability);
    }
}

int main() {
    // Generate points (Mt vs Mh grid)
    vector<pair<double,double>> points;
    // Increased range to ensure we capture the full phase space

    // Also add a coarser wide grid just in case
    for (double Mt = 0; Mt <= 300; Mt += 2.0) {
        for (double Mh = 0; Mh <= 300; Mh += 2.0) {
            points.emplace_back(Mt, Mh);
        }
    }
    
    cout << "Analyzing " << points.size() << " points with NNLO matching...\n";
    
    vector<tuple<double,double,double,int>> results(points.size());
    process_chunk(points, results, 0, points.size());
    
    ofstream file("../../results/data.csv");
    file << "Mt,Calculated_Mh,Input_Mh,Stability\n";
    for (size_t i = 0; i < results.size(); ++i) {
        double Mt = std::get<0>(results[i]);
        double calculated_mh = std::get<1>(results[i]);
        double Mh = std::get<2>(results[i]);
        int st = std::get<3>(results[i]);
        file << Mt << "," << calculated_mh << "," << Mh << "," << st << "\n";
    }
    file.close();
    
    cout << "\nData saved to data.csv\n";
    cout << "Stability codes: 1=Stable, 2=Metastable, 3=Unstable, 4=Invalid\n";
    
    return 0;
}