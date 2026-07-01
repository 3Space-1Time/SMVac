import pandas as pd
import numpy as np

df = pd.read_csv('data/max_steps_diag.csv')
print('Total rows:', len(df))
print(df.head())
print('----------------')
print(df.tail(10))

print('\n--- Trajectory Characteristics at MAX_STEPS ---')
last = df.iloc[-1]
print(f"r at 100k steps: {last['r']:.4e}")
print(f"phi at 100k steps: {last['phi']:.4e}")
print(f"dphi at 100k steps: {last['dphi']:.4e}")
print(f"dV at 100k steps: {last['dV']:.4e}")
print(f"dr at 100k steps: {last['dr']:.4e}")
print(f"Truncation error: {last['truncation_error']:.4e}")
print(f"Friction term: {last['friction']:.4e}")

try:
    ratio = abs(last['friction'] / (last['dV'] + 1e-100))
    print(f"Ratio of friction to dV: {ratio:.4e}")
except:
    pass
