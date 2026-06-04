import re

with open('bounce_action_solver.cpp', 'r') as f:
    code = f.read()

# Replace RGEHelper
old_rge = '''class RGEHelper {
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
};'''

code = re.sub(r'class RGEHelper \{.*?\};', old_rge, code, flags=re.DOTALL)

# Replace rge.build_fast_lookup()
code = code.replace('rge.build_fast_lookup();', '')

# Replace evaluate_action_at_R
old_eval = '''double evaluate_action_at_R(RGEHelper& rge, double mu_inst, double R) {
    double t_R = 2.0 * log(1.0/R);
    Params p_R = rge.get_params(t_R);
    
    if (p_R.lambda >= 0) {
        return 1e9; 
    }
    
    double prefactor = sqrt(2.0 / std::abs(p_R.lambda));
    double kinetic_term = (16.0 * pi2) / (3.0 * std::abs(p_R.lambda));
    
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
        else if (i % 2 != 0) weight = 4.0 / 3.0;
        else weight = 2.0 / 3.0;
        
        potential_integral += weight * integrand * dx;
    }
    
    return kinetic_term + potential_integral;
}'''

code = re.sub(r'double evaluate_action_at_R\(RGEHelper& rge, double mu_inst, double R\).*?return kinetic_term \+ potential_integral;\n\}', old_eval, code, flags=re.DOTALL)

with open('bounce_action_solver_unopt.cpp', 'w') as f:
    f.write(code)

print("Reverted to unoptimized version for testing")
