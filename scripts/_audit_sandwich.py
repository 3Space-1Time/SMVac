import pandas as pd
import numpy as np
import sys

df = pd.read_csv('data/numerical_data.csv')

# --- 1. Count negative S_exact values ---
neg = df[df['S_exact'] < 0]
print(f"Total negative S_exact: {len(neg)} out of {len(df)} ({100*len(neg)/len(df):.3f}%)")
if len(neg) > 0:
    min_s = neg['S_exact'].min()
    print(f"Min S_exact: {min_s:.1f}")
    print(f"M_t range: [{neg['Mt'].min():.1f}, {neg['Mt'].max():.1f}]")
    print(f"M_h range: [{neg['Mh_calc'].min():.1f}, {neg['Mh_calc'].max():.1f}]")
    print()

# --- 2. Check sandwiched region near M_h ~ 50 GeV ---
mask = (df['Mh_calc'] >= 40) & (df['Mh_calc'] <= 70)
sub = df[mask].copy()

print("=== Sandwiched region analysis (M_h 40-70) ===")
found = False
for mh in sorted(sub['Mh_calc'].unique()):
    chunk = sub[sub['Mh_calc'] == mh].sort_values('Mt')
    stabs = chunk[['Mt', 'Stability', 'S_exact']].values
    pattern = ''.join(str(int(row[1])) for row in stabs)
    neg_in = chunk[chunk['S_exact'] < 0]
    if '3' in pattern[:10] or len(neg_in) > 0:
        found = True
        print(f"M_h = {mh:.1f}: pattern={pattern[:50]}...")
        print(f"  M_t: {stabs[0][0]:.1f} to {stabs[-1][0]:.1f}")
        if len(neg_in) > 0:
            print(f"  Negative S: {len(neg_in)} points, min S={neg_in['S_exact'].min():.1f}")

if not found:
    print("No instability patterns found in this range.")

# --- 3. Check for S_exact = -1 (sentinels) ---
sentinel = df[df['S_exact'] == -1.0]
print(f"\nPoints with S_exact = -1 (sentinels): {len(sentinel)}")
if len(sentinel) > 0:
    print(f"  Stability values: {sentinel['Stability'].value_counts().to_dict()}")

# --- 4. Check S_exact values near zero ---
near_zero = df[(df['S_exact'] > -100) & (df['S_exact'] < 100) & (df['Stability'] != 4)]
print(f"\nPoints with |S_exact| < 100 (non-nonpert): {len(near_zero)}")
print(f"  Min S: {near_zero['S_exact'].min():.1f}, Max S: {near_zero['S_exact'].max():.1f}")
neg_near = near_zero[near_zero['S_exact'] < 0]
print(f"  Negative S in this set: {len(neg_near)} points")
if len(neg_near) > 0:
    print(f"  Their (M_t, M_h, S):")
    for _, row in neg_near.iterrows():
        print(f"    M_t={row['Mt']:.1f}, M_h={row['Mh_calc']:.1f}, S={row['S_exact']:.1f}")
