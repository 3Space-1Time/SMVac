with open('c:/Users/LENOVO/code/Threshold/src/sm_stability/vacuum_precision_test_quad.cpp', 'r') as f:
    text = f.read()

text = text.replace('\\n', '\n')
text = text.replace('double get_energy_at(double phi_val) {\n        double phi_arr[] = {phi_val};\n        return vpot(phi_arr);\n    }', 'double get_energy_at(double phi_val) {\n        qFloat phi_arr[] = {(qFloat)phi_val};\n        return (double)vpot(phi_arr);\n    }')

with open('c:/Users/LENOVO/code/Threshold/src/sm_stability/vacuum_precision_test_quad.cpp', 'w') as f:
    f.write(text)

