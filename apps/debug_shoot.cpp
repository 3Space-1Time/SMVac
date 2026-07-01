#include <iostream>
#include <cmath>
#include <vector>
#include "solver_numerical.cpp"

struct PotFD {
    std::vector<double> V_vals, dV_vals;
    double min_log_phi, d_log;
    double min_phi;
    
    double V_eff_eval(RGEHelper& rge, double phi) {
        if (phi <= v) return 0.0;
        double t = 2.0 * std::log(phi);
        auto p = rge.get_params(t);
        double lambda = rge.get_lambda_eff(p, phi, false);
        return 0.25 * lambda * std::pow(phi, 4);
    }
    
    double dV_fd(RGEHelper& rge, double phi) {
        if (phi <= v) return 0.0;
        double h = phi * 1e-4;
        return (V_eff_eval(rge, phi - 2*h) - 8*V_eff_eval(rge, phi - h) + 8*V_eff_eval(rge, phi + h) - V_eff_eval(rge, phi + 2*h)) / (12.0 * h);
    }
    
    void init(RGEHelper& rge, double min_p, double max_p, int N) {
        min_phi = min_p;
        min_log_phi = std::log(min_p);
        double max_log_phi = std::log(max_p);
        d_log = (max_log_phi - min_log_phi) / (N - 1);
        V_vals.resize(N); dV_vals.resize(N);
        for (int i=0; i<N; ++i) {
            double phi = std::exp(min_log_phi + i * d_log);
            V_vals[i] = V_eff_eval(rge, phi);
            dV_vals[i] = dV_fd(rge, phi);
        }
    }
    
    void eval(double phi, double& V, double& dV) const {
        if (phi <= min_phi) { V=0; dV=0; return; }
        double lphi = std::log(phi);
        double idx_d = (lphi - min_log_phi) / d_log;
        if (idx_d >= V_vals.size() - 1) {
            V = V_vals.back(); dV = dV_vals.back(); return;
        }
        int idx = (int)idx_d;
        double frac = idx_d - idx;
        V = V_vals[idx] + frac * (V_vals[idx+1] - V_vals[idx]);
        dV = dV_vals[idx] + frac * (dV_vals[idx+1] - dV_vals[idx]);
    }
};

struct State { double phi, dphi; };

double shoot_debug(const PotFD& pot, double phi_0) {
    State y = {phi_0, 0.0};
    double r = 1e-4;
    double max_r = 1e8;
    int step_cnt = 0;
    
    while (r < max_r) {
        if (!std::isfinite(y.phi) || !std::isfinite(y.dphi)) {
            std::cout << "NaN hit at r=" << r << ", phi=" << y.phi << ", dphi=" << y.dphi << "\n";
            return -1.0;
        }
        if (y.phi < 100.0 && y.dphi < 0) {
            std::cout << "Overshoot! Crossed v at r=" << r << "\n";
            return y.phi;
        }
        if (y.dphi > 0) {
            std::cout << "Undershoot! Turned around at r=" << r << ", phi=" << y.phi << "\n";
            return y.phi;
        }
        if (y.phi < 0) {
            std::cout << "Crossed 0 at r=" << r << "\n";
            return y.phi;
        }
        
        double dr_target = 0.01 * std::abs(y.phi / (y.dphi + 1e-30));
        double dr = std::max(1e-5, std::min(dr_target, 1.0));
        
        auto derivs = [&](double curr_r, State curr_y) -> State {
            double V, dV;
            pot.eval(curr_y.phi, V, dV);
            double friction = (curr_r == 0) ? 0 : (3.0 / curr_r) * curr_y.dphi;
            return {curr_y.dphi, dV - friction};
        };
        
        State k1 = derivs(r, y);
        State k2 = derivs(r + 0.5*dr, {y.phi + 0.5*dr*k1.phi, y.dphi + 0.5*dr*k1.dphi});
        State k3 = derivs(r + 0.5*dr, {y.phi + 0.5*dr*k2.phi, y.dphi + 0.5*dr*k2.dphi});
        State k4 = derivs(r + dr, {y.phi + dr*k3.phi, y.dphi + dr*k3.dphi});
        
        y.phi += (dr/6.0)*(k1.phi + 2*k2.phi + 2*k3.phi + k4.phi);
        y.dphi += (dr/6.0)*(k1.dphi + 2*k2.dphi + 2*k3.dphi + k4.dphi);
        r += dr;
        
        step_cnt++;
        if (step_cnt % 1000000 == 0) {
            std::cout << "Step " << step_cnt << ": r=" << r << ", phi=" << y.phi << ", dphi=" << y.dphi << ", dr=" << dr << "\n";
        }
    }
    std::cout << "Max r reached! phi=" << y.phi << ", dphi=" << y.dphi << "\n";
    return y.phi;
}

int main() {
    RGEHelper rge;
    auto p_ew = get_nnlo_matching(5.0, 105.0);
    rge.solve(p_ew, 2*std::log(100.0), 2*std::log(1.22e19), 0.1);
    
    PotFD pot;
    std::cout << "Init pot...\n";
    pot.init(rge, 100.0, 1e18, 1000000);
    
    std::cout << "Testing shoot at 1e5...\n";
    shoot_debug(pot, 1e5);
    std::cout << "Testing shoot at 1e6...\n";
    shoot_debug(pot, 1e6);
    std::cout << "Testing shoot at 1e8...\n";
    shoot_debug(pot, 1e8);
    
    return 0;
}
