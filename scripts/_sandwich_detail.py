import pandas as pd, numpy as np, sys

df = pd.read_csv('data/numerical_data.csv')

# ---- Detailed S_exact profiling at M_h = 5, 50, 100 ----
for mh_label, mh_target, tol in [('Mh=5', 5.0, 0.02), ('Mh=50', 50.0, 0.02), ('Mh=100', 100.0, 0.02)]:
    chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt').copy()
    n = len(chunk)
    median_mh = chunk['Mh_calc'].median()
    print(f"\n=== {mh_label} (Mh={median_mh:.3f}, N={n} pts) ===")
    print(f"{'Mt':>8} {'Stab':>4} {'S_exact':>12} {'S_approx':>12} {'S_ratio':>10} {'dS_%':>8}")
    print('-' * 60)
    for _, row in chunk.iterrows():
        S_ex = row['S_exact']
        S_ap = row['S_approx']
        ratio = S_ex / S_ap if S_ap > 0 else 0
        dS = 100 * (S_ex - S_ap) / S_ex if S_ex > 0 else 0
        print(f"{row['Mt']:>8.2f} {int(row['Stability']):>4} {S_ex:>12.4f} {S_ap:>12.4f} {ratio:>10.4f} {dS:>8.2f}")
    # Also show sandwiched region boundaries
    stabs = chunk['Stability'].values.astype(int)
    for i in range(1, len(stabs)):
        if stabs[i] != stabs[i-1]:
            mt_here = chunk.iloc[i]['Mt']
            print(f"  >> Transition at Mt={mt_here:.2f}: Stab {stabs[i-1]} -> {stabs[i]}")
