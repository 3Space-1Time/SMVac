import re

with open('bounce_action_solver.cpp', 'r') as f:
    code = f.read()

# Add get_params back to RGEHelper
get_params_code = '''
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
'''

code = code.replace('double get_lambda_fast(double t) const {', get_params_code + '\n    double get_lambda_fast(double t) const {')

# Fix min_lambda in evaluate_action_at_R
code = code.replace('double phi = sqrt(2.0 / std::abs(min_lambda)) * (2.0 * R) / (r*r + R*R);', 'double phi = sqrt(2.0 / std::abs(lambda_R)) * (2.0 * R) / (r*r + R*R);')
code = code.replace('return (16.0 * pi2) / (3.0 * std::abs(min_lambda)) + S;', 'return (16.0 * pi2) / (3.0 * std::abs(lambda_R)) + S;')

with open('bounce_action_solver.cpp', 'w') as f:
    f.write(code)

print("Fixed missing references")
