import pandas as pd
import numpy as np

df = pd.read_csv('data/numerical_data.csv')

# Only consider points where S_exact was actually computed (not -1 sentinel)
computed = df[df['S_exact'] != -1.0].copy()

print(f"Total computed points: {len(computed)}")
print(f"Stable (1): {len(computed[computed['Stability']==1])}")
print(f"Metastable (2): {len(computed[computed['Stability']==2])}")
print(f"Unstable (3): {len(computed[computed['Stability']==3])}")
print(f"Non-pert (4): {len(computed[computed['Stability']==4])}")

# --- Negative S_exact ---
neg = computed[computed['S_exact'] < 0]
print(f"\n=== Negative S_exact ===")
print(f"Count: {len(neg)}")
if len(neg) > 0:
    print(f"Min: {neg['S_exact'].min():.3f}")
    print(f"Max (negative): {neg['S_exact'].max():.3f}")
    print(f"M_t range: [{neg['Mt'].min():.1f}, {neg['Mt'].max():.1f}]")
    print(f"M_h range: [{neg['Mh_calc'].min():.1f}, {neg['Mh_calc'].max():.1f}]")
    print(f"Stability values: {neg['Stability'].value_counts().to_dict()}")
    
    # Look at the most negative points
    print("\nMost negative S_exact (top 20):")
    most_neg = neg.nsmallest(20, 'S_exact')
    for _, row in most_neg.iterrows():
        print(f"  M_t={row['Mt']:.1f}, M_h={row['Mh_calc']:.1f}, S={row['S_exact']:.1f}, Stab={int(row['Stability'])}")

# --- Sandwiched region at M_h ~ 50 GeV ---
print(f"\n=== Sandwiched region: M_h near 50 GeV ===")
for mh_target in [45, 50, 55, 60]:
    tol = 0.3
    chunk = computed[(computed['Mh_calc'] >= mh_target - tol) & 
                     (computed['Mh_calc'] <= mh_target + tol)].sort_values('Mt')
    if len(chunk) == 0:
        continue
    stabs = chunk[['Mt', 'Stability', 'S_exact']].values
    pattern = ''.join(str(int(row[1])) for row in stabs)
    neg_in = chunk[chunk['S_exact'] < 0]
    print(f"\nM_h ~ {mh_target}: {len(chunk)} points, pattern={pattern[:80]}...")
    print(f"  M_t: {stabs[0][0]:.1f} to {stabs[-1][0]:.1f}")
    if len(neg_in) > 0:
        print(f"  Negative S: {len(neg_in)} pts, range [{neg_in['S_exact'].min():.1f}, {neg_in['S_exact'].max():.1f}]")

# --- Check: do Stability=3 points have negative S_exact? ---
unstable = computed[computed['Stability'] == 3]
print(f"\n=== Stability=3 (Unstable) points ===")
print(f"Total unstable: {len(unstable)}")
print(f"Of which S_exact < 0: {len(unstable[unstable['S_exact'] < 0])}")
print(f"Of which S_exact >= 0: {len(unstable[unstable['S_exact'] >= 0])}")
if len(unstable) > 0:
    print(f"S_exact min={unstable['S_exact'].min():.3f}, max={unstable['S_exact'].max():.3f}")
    pos_unstable = unstable[unstable['S_exact'] >= 0]
    print(f"Positive S unstable: count={len(pos_unstable)}, range [{pos_unstable['S_exact'].min():.3f}, {pos_unstable['S_exact'].max():.3f}]")
