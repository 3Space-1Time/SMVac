import re

with open('bounce_action_solver.cpp', 'r') as f:
    code = f.read()

# Replace the entire grid generation in main
new_grid = '''    vector<pair<double,double>> points;
    
    // Single dense uniform grid for beautiful rendering
    double dM = 0.25;
    for(double Mh = 10; Mh <= 250; Mh += dM) {
        for(double Mt = 100; Mt <= 250; Mt += dM) {
            points.emplace_back(Mt, Mh);
        }
    }
    
    // Deduplicate points
    sort(points.begin(), points.end());
    points.erase(unique(points.begin(), points.end()), points.end());'''

code = re.sub(r'    vector<pair<double,double>> points;.*?points\.erase\(unique\(points\.begin\(\), points\.end\(\)\), points\.end\(\)\);', new_grid, code, flags=re.DOTALL)

with open('bounce_action_solver.cpp', 'w') as f:
    f.write(code)

print("Grid generation replaced.")
