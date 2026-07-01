import os

code = """
#include <iostream>
#include <cmath>
#include <vector>
#include "solver_numerical.cpp" // to get RGEHelper, etc.

double V_eff(RGEHelper& rge, double phi) {
    if (phi <= v) return 0.0;
    double t = 2.0 * std::log(phi);
    auto p = rge.get_params(t);
    double lambda = rge.get_lambda_eff(p, phi, false);
    double c6_val = 0.0; // Assume 0 for this test
    double v_tree = 0.25 * lambda * std::pow(phi, 4);
    double v_c6 = c6_val * std::pow(phi, 6) / std::pow(M_PL, 2);
    return v_tree + v_c6;
}

double dV_fd(RGEHelper& rge, double phi) {
    if (phi <= v) return 0.0;
    double h = phi * 1e-4;
    double v_m2 = V_eff(rge, phi - 2*h);
    double v_m1 = V_eff(rge, phi - h);
    double v_p1 = V_eff(rge, phi + h);
    double v_p2 = V_eff(rge, phi + 2*h);
    return (v_m2 - 8*v_m1 + 8*v_p1 - v_p2) / (12.0 * h);
}

int main() {
    RGEHelper rge;
    auto p_ew = get_nnlo_matching(5.0, 105.0);
    rge.solve(p_ew, 2*std::log(v), 2*std::log(1.22e19), 0.1);
    
    std::cout << "phi = 100, dV = " << dV_fd(rge, 100.0) << std::endl;
    std::cout << "phi = 1e5, dV = " << dV_fd(rge, 1e5) << std::endl;
    std::cout << "phi = 1e8, dV = " << dV_fd(rge, 1e8) << std::endl;
    std::cout << "phi = 1e18, dV = " << dV_fd(rge, 1e18) << std::endl;
    return 0;
}
"""

with open("c:/Users/LENOVO/code/Threshold/test_nan.cpp", "w") as f:
    f.write(code)
