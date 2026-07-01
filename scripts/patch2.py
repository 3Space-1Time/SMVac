import os
import subprocess
import pandas as pd
import numpy as np

# Just the fix to run_derivative_investigation.py: add isfinite check in shoot!
with open('c:/Users/LENOVO/.gemini/antigravity-ide/brain/48e2c0f8-de74-4cf9-a77c-2f56d66e271a/scratch/run_derivative_investigation.py', 'r') as f:
    text = f.read()

# Add isfinite check
text = text.replace('if (y.phi < v && y.dphi < 0) return y.phi;', 
                    'if (!std::isfinite(y.phi) || !std::isfinite(y.dphi)) return -1.0;\n        if (y.phi < v && y.dphi < 0) return y.phi;')

# Also, if we return -1.0, it means failure to overshoot. The loop in find_root needs to handle res == -1.0
text = text.replace('if (res < v) { found_overshoot = true; break; }',
                    'if (res < v && res != -1.0) { found_overshoot = true; break; }\n        if (res == -1.0) { break; }')

with open('c:/Users/LENOVO/.gemini/antigravity-ide/brain/48e2c0f8-de74-4cf9-a77c-2f56d66e271a/scratch/run_derivative_investigation.py', 'w') as f:
    f.write(text)
