import re

with open('vacuum_precision_test.cpp', 'r') as f:
    text = f.read()

physics_body = text.split('tuple<double, double> calculate_precision')[0]

main_body = '''
double phi_true_guess = -1.0;

int main() {
    double Mt = 180.0;
    double Mh = 120.0;
    double lambda0 = 0.12604 + 0.00206*(Mh - 125.15) - 0.00004*(Mt - 173.34);
    double yt0 = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alpha3_at_Mz - 0.1184) / 0.0007;
    
    double t0 = 2*log(172.5);
    double tPlanck = 2*log(MPlanck);
    
    Params y = {g1init, g2init, g3init, yt0, sqrt(2)*Mb/v, sqrt(2)*Mtau/v, lambda0, 0};
    RGEHelper rge;
    double mu_inst = MPlanck;
    
    double t = t0;
    double dt = 0.1;
    rge.add_point(t, y);
    bool is_unstable = false;
    
    while (t < tPlanck) {
        if (t + dt > tPlanck) dt = tPlanck - t;
        if (!rk4_adaptive_step(y, t, dt)) {
            break;
        }
        rge.add_point(t, y);
        if (!is_unstable && y.lambda < 0) {
            mu_inst = exp(t/2);
            is_unstable = true;
        }
    }
    
    SMPotential pot(rge, mu_inst);
    double min_energy = 0.0;
    
    for (double p = 1.0; p < 10000.0; p *= 1.05) {
        double e = pot.get_energy_at(p);
        if (e < min_energy) { 
            min_energy = e;
            phi_true_guess = p;
        }
    }
    
    cout << "--- Discretization Error Test ---" << endl;
    ofstream f_disc("../../results/discretization_error.csv");
    f_disc << "N,S_exact\\n";
    
    vector<int> N_vals = {100};
    for (int N : N_vals) {
        SB::BounceCalculator bounce;
        bounce.verboseOn();
        bounce.setModel(&pot);
        double phi_true_arr[] = {phi_true_guess}; 
        double phi_false_arr[] = {0.0};
        bounce.setVacuum(phi_true_arr, phi_false_arr);
        bounce.setMaxN(100000); 
        bounce.setN(N);
        bounce.solve();
        f_disc << N << "," << bounce.action() << "\\n";
        cout << "N = " << N << " -> S = " << bounce.action() << endl;
    }
    f_disc.close();
    
    cout << "--- Finite Volume Error Test ---" << endl;
    ofstream f_vol("../../results/finite_volume_error.csv");
    f_vol << "Rmax,S_exact\\n";
    
    double dr = 0.001;
    vector<double> R_vals = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0};
    for (double R : R_vals) {
        int N = max(10, (int)(R / dr));
        SB::BounceCalculator bounce;
        bounce.setModel(&pot);
        double phi_true_arr[] = {phi_true_guess}; 
        double phi_false_arr[] = {0.0};
        bounce.setVacuum(phi_true_arr, phi_false_arr);
        bounce.setMaxN(100000);
        bounce.setRmax(R);
        bounce.setN(N);
        bounce.solve();
        f_vol << R << "," << bounce.action() << "\\n";
        cout << "Rmax = " << R << " (N=" << N << ") -> S = " << bounce.action() << endl;
    }
    f_vol.close();
    
    return 0;
}
'''
with open('error_scaling_test2.cpp', 'w') as f:
    f.write(physics_body + main_body)
