import re
with open('bounce_action_solver.cpp', 'r') as f:
    code = f.read()

code = re.sub(r'int main\(\) \{.*', '''int main() {
    double Mh = 230.0;
    double Mt = 50.0;
    
    double lambda0 = Mh*Mh / (2*v*v);
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    double t0 = 2*log(172.5);
    double tPlanck = 2*log(MPlanck);
    
    Params y = {g1init, g2init, g3init, yt0, sqrt(2.0)*Mb/v, sqrt(2.0)*Mtau/v, lambda0, 0};
    
    double t = t0;
    double dt = 0.1;
    
    std::cout << "t0 = " << t0 << ", lambda0 = " << lambda0 << ", yt0 = " << yt0 << std::endl;
    
    int steps = 0;
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        if (!rk4_adaptive_step(y, t, dt)) {
            std::cout << "Pole hit (rk4 false) at t=" << t << ", lambda=" << y.lambda << std::endl;
            return 0;
        }
        steps++;
        if (steps % 50 == 0) std::cout << "t=" << t << ", lambda=" << y.lambda << ", yt=" << y.yt << std::endl;
        
        if (std::abs(y.lambda) > 4*pi || y.yt > 4*pi) {
            std::cout << "Exceeded 4pi at t=" << t << ", lambda=" << y.lambda << std::endl;
            return 0;
        }
    }
    std::cout << "Reached tPlanck! lambda=" << y.lambda << std::endl;
    return 0;
}
''', code, flags=re.DOTALL)

with open('test_solver.cpp', 'w') as f:
    f.write(code)
