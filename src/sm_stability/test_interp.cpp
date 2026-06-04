#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

struct Params {
    double g1, g2, g3, yt, yb, ytau, lambda, phi;
};

class RGEHelper {
private:
    vector<pair<double, Params>> table;
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
};

int main() {
    RGEHelper rge;
    for (int i = 0; i < 1000; i++) {
        double t = 10.0 + i * 0.078;
        Params p = {0};
        p.lambda = sin(t); // arbitrary function
        rge.add_point(t, p);
    }
    rge.build_fast_lookup();
    
    double max_err = 0;
    for (int i = 0; i < 5000; i++) {
        double t = 10.0 + i * 0.015;
        double l1 = rge.get_params(t).lambda;
        double l2 = rge.get_lambda_fast(t);
        max_err = max(max_err, abs(l1 - l2));
    }
    cout << "Max error between fast and slow: " << max_err << endl;
    return 0;
}
