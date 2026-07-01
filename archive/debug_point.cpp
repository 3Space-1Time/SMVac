#include <iostream>
#include <cmath>

using namespace std;

const double v = 246.22;
const double alpha3_at_Mz = 0.1184; 
const double pi = 3.14159265358979323846;
const double PI2 = pi * pi;
const double LOOP1 = 16.0 * PI2;
const double LOOP2 = LOOP1 * LOOP1;
const double LOOP3 = LOOP2 * LOOP1;
const double LOOP4 = LOOP2 * LOOP2;

struct Params { 
    double g1, g2, g3, yt, yb, ytau, lambda, phi; 
};

// ... copy the functions
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
    const double Mb = 4.0;
    const double Mtau = 1.777;
    return {g1_Mt, g2_Mt, g3_Mt, yt_Mt, sqrt(2.0)*Mb/v, sqrt(2.0)*Mtau/v, lambda_Mt, 0};
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

double betaG1sq(const Params& p) { return 0; } // placeholder for compilation if not needed
double betaG2sq(const Params& p) { return 0; }
double betaG3sq(const Params& p) { return 0; }
double betaYt(const Params& p) { return 0; }
double betaYb(const Params& p) { return 0; }
double betaYtau(const Params& p) { return 0; }

double betaLambda(const Params& p) {
    double g1_2 = p.g1*p.g1, g2_2 = p.g2*p.g2;
    double yt2 = p.yt*p.yt, yb2 = p.yb*p.yb, ytau2 = p.ytau*p.ytau;
    double lam = p.lambda;
    
    double term1 = 1.0/LOOP1 * (
        24*lam*lam - 3*lam*(g1_2 + 3*g2_2) + 3.0/8.0*(g1_2*g1_2 + 2*g1_2*g2_2 + 3*g2_2*g2_2)
        + 4*lam*(3*yt2 + 3*yb2 + ytau2) - 2*(3*yt2*yt2 + 3*yb2*yb2 + ytau2*ytau2)
    );
    // neglecting loop 2 and 3 for this simple debug printing to save code
    return term1;
}

// simple rk4 step
Params rk4_single_step(Params y, double t, double dt) {
    auto dydt = [](const Params& p) {
        Params dy;
        dy.g1 = 0; dy.g2 = 0; dy.g3 = 0;
        dy.yt = 0; dy.yb = 0; dy.ytau = 0; // simple beta functions for debug
        dy.lambda = betaLambda(p); dy.phi = 0;
        return dy;
    };
    Params k1 = dydt(y);
    Params k2 = dydt({y.g1+0.5*dt*k1.g1, y.g2+0.5*dt*k1.g2, y.g3+0.5*dt*k1.g3, y.yt+0.5*dt*k1.yt, y.yb+0.5*dt*k1.yb, y.ytau+0.5*dt*k1.ytau, y.lambda+0.5*dt*k1.lambda, 0});
    Params k3 = dydt({y.g1+0.5*dt*k2.g1, y.g2+0.5*dt*k2.g2, y.g3+0.5*dt*k2.g3, y.yt+0.5*dt*k2.yt, y.yb+0.5*dt*k2.yb, y.ytau+0.5*dt*k2.ytau, y.lambda+0.5*dt*k2.lambda, 0});
    Params k4 = dydt({y.g1+dt*k3.g1, y.g2+dt*k3.g2, y.g3+dt*k3.g3, y.yt+dt*k3.yt, y.yb+dt*k3.yb, y.ytau+dt*k3.ytau, y.lambda+dt*k3.lambda, 0});
    
    Params next_y = y;
    next_y.lambda += dt/6*(k1.lambda+2*k2.lambda+2*k3.lambda+k4.lambda);
    return next_y;
}

int main() {
    double Mh = 2.0;
    double Mt = 50.0;
    
    Params y = get_nnlo_matching(Mh, Mt);
    cout << "Initial matching at Mt = " << Mt << " Mh = " << Mh << endl;
    cout << "g1=" << y.g1 << " g2=" << y.g2 << " g3=" << y.g3 << endl;
    cout << "yt=" << y.yt << " lambda_tree=" << (Mh*Mh)/(2.0*v*v) << " lambda_Mt=" << y.lambda << endl;
    
    double min_lambda_eff = get_lambda_eff(y);
    cout << "Initial lambda_eff=" << min_lambda_eff << endl;
    
    double t = 2*log(Mt);
    double dt = 0.1;
    double tMax = 2*log(1.22e19);
    
    bool went_negative = false;
    double mu1 = -1.0;
    if (min_lambda_eff <= 0) {
        went_negative = true;
        mu1 = exp(t/2.0);
        cout << "Negative at start! mu1=" << mu1 << endl;
    }
    
    int step = 0;
    while (t < tMax) {
        if (t + dt > tMax) dt = tMax - t;
        y = rk4_single_step(y, t, dt);
        t += dt;
        double lam_eff = get_lambda_eff(y);
        
        if (lam_eff < min_lambda_eff) min_lambda_eff = lam_eff;
        if (!went_negative && lam_eff <= 0.0) {
            mu1 = std::exp(t / 2.0);
            went_negative = true;
            cout << "Went negative at t=" << t << " mu1=" << mu1 << " lam_eff=" << lam_eff << endl;
        }
        if (step++ < 5) cout << "step " << step << " t=" << t << " lambda=" << y.lambda << " lam_eff=" << lam_eff << endl;
    }
    cout << "min_lambda_eff = " << min_lambda_eff << endl;
    
    double Tv = 1.179e44 / v;
    double S_approx = 8.0 * pi * pi / (3.0 * std::abs(min_lambda_eff));
    double S_threshold = 4.0 * std::log(Tv * mu1);
    cout << "S_approx = " << S_approx << endl;
    cout << "S_threshold = " << S_threshold << " (using mu1=" << mu1 << ")" << endl;
    if (S_approx > S_threshold) cout << "Result: Metastable (2)" << endl;
    else cout << "Result: Unstable (3)" << endl;
    
    return 0;
}
