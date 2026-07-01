import re

with open('c:/Users/LENOVO/.gemini/antigravity-ide/brain/48e2c0f8-de74-4cf9-a77c-2f56d66e271a/scratch/run_derivative_investigation.py', 'r') as f:
    text = f.read()

text = text.replace('if (res < v)', 'std::cout << "phi_high=" << phi_high << " res=" << res << "\\n";\\n        if (res < v)')

# also fix the rel_err bug
text = text.replace("df_deriv['rel_err'] = np.abs(df_deriv['dV_fd'] - df_deriv['dV_analytic']) / np.abs(df_deriv['dV_analytic'] + 1e-30)",
                    "df_deriv['rel_err'] = np.abs(df_deriv['dV_fd'] - df_deriv['dV_analytic']) / np.abs(df_deriv['dV_analytic'] + 1e-30) + 1e-15")

with open('c:/Users/LENOVO/.gemini/antigravity-ide/brain/48e2c0f8-de74-4cf9-a77c-2f56d66e271a/scratch/run_derivative_investigation.py', 'w') as f:
    f.write(text)
