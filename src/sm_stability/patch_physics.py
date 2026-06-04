import re

with open('bounce_action_solver.cpp', 'r') as f:
    code = f.read()

# I will replace the entire evaluate_action_at_R function with the exact original one, 
# just swapping rge.get_params(t).lambda to rge.get_lambda_fast(t).
original_func = '''double evaluate_action_at_R(RGEHelper& rge, double mu_inst, double R) {
    double t_R = 2.0 * log(1.0/R);
    double lambda_R = rge.get_lambda_fast(t_R);
    
    if (lambda_R >= 0) {
        return 1e9; 
    }
    
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
            double lam = rge.get_lambda_fast(t);
            V_x = 0.25 * lam * pow(phi_x, 4);
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

# Replace the broken evaluate_action_at_R
code = re.sub(r'double evaluate_action_at_R\(RGEHelper& rge, double mu_inst, double R\).*?return.*?\}', original_func, code, flags=re.DOTALL)

with open('bounce_action_solver.cpp', 'w') as f:
    f.write(code)

print("Restored exact physics with fast lambda!")
