import re

with open('bounce_action_solver.cpp', 'r') as f:
    code = f.read()

replacement = '''class RGEHelper {
private:
    vector<pair<double, Params>> table;
    
    // Fast lookup members
    vector<double> lambda_fast;
    double t_min_fast = 0;
    double t_max_fast = 0;
    double dt_fast = 0;
    double inv_dt_fast = 0;

public:
    void clear() { 
        table.clear(); 
        lambda_fast.clear();
    }
    void add_point(double t, const Params& p) { table.push_back({t, p}); }
    
    void build_fast_lookup() {
        if (table.empty()) return;
        t_min_fast = table.front().first;
        t_max_fast = table.back().first;
        int N_bins = 10000;
        dt_fast = (t_max_fast - t_min_fast) / N_bins;
        if (dt_fast <= 0) {
            lambda_fast.push_back(table.front().second.lambda);
            return;
        }
        inv_dt_fast = 1.0 / dt_fast;
        lambda_fast.resize(N_bins + 1);
        
        int current_idx = 0;
        for (int i = 0; i <= N_bins; ++i) {
            double t = t_min_fast + i * dt_fast;
            while (current_idx < table.size() - 1 && table[current_idx + 1].first < t) {
                current_idx++;
            }
            if (current_idx >= table.size() - 1) {
                lambda_fast[i] = table.back().second.lambda;
            } else {
                double t1 = table[current_idx].first;
                double t2 = table[current_idx + 1].first;
                double l1 = table[current_idx].second.lambda;
                double l2 = table[current_idx + 1].second.lambda;
                lambda_fast[i] = l1 + (t - t1) / (t2 - t1) * (l2 - l1);
            }
        }
    }
    
    double get_lambda_fast(double t) const {
        if (lambda_fast.empty()) return 0;
        if (t <= t_min_fast) return lambda_fast.front();
        if (t >= t_max_fast) return lambda_fast.back();
        
        double bin = (t - t_min_fast) * inv_dt_fast;
        int idx = (int)bin;
        if (idx >= lambda_fast.size() - 1) return lambda_fast.back();
        
        double frac = bin - idx;
        return lambda_fast[idx] + frac * (lambda_fast[idx+1] - lambda_fast[idx]);
    }
};'''

code = re.sub(r'class RGEHelper \{.*?\};', replacement, code, flags=re.DOTALL)

# Add build_fast_lookup call in classify_stability
# Find where rge finishes adding points, before action evaluation.
# Wait, classify_stability returns std::tuple right after while loop if it's stable.
# But for unstable it proceeds to evaluate_action_at_R.
# So we can just add ge.build_fast_lookup(); right before evaluate_action_at_R
code = code.replace('return evaluate_action_at_R(rge, mu_inst, min_R);', 'rge.build_fast_lookup();\n    return evaluate_action_at_R(rge, mu_inst, min_R);')

# Update evaluate_action_at_R to use get_lambda_fast
code = code.replace('double lam = rge.get_params(t).lambda;', 'double lam = rge.get_lambda_fast(t);')

with open('bounce_action_solver.cpp', 'w') as f:
    f.write(code)
print("Optimization patched.")
