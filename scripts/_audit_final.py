import pandas as pd, numpy as np

df = pd.read_csv('data/numerical_data.csv')
print(f"Total points: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nAll points by Stability:")
print(df['Stability'].value_counts().sort_index().to_dict())

# Are sentinel points Stab=1 or Stab=4?
sentinel = df[df['S_exact'] == -1.0]
computed = df[df['S_exact'] != -1.0]
print(f"\n=== Sentinel (S_exact=-1) ===")
print(f"Count: {len(sentinel)}")
print(f"Stability values: {sentinel['Stability'].value_counts().to_dict()}")

print(f"\n=== Computed (S_exact != -1) ===")
print(f"Count: {len(computed)}")  
print(f"Stability values: {computed['Stability'].value_counts().to_dict()}")

# Check S_approx vs S_exact on computed points
match = np.isclose(computed['S_exact'], computed['S_approx'], rtol=0.01)
print(f"\nS_exact vs S_approx match within 1%: {match.sum()} / {len(computed)} ({100*match.sum()/len(computed):.1f}%)")

# Also check the approximate region near experimental point
exp = df[(df['Mh_calc'] >= 125.0) & (df['Mh_calc'] <= 125.2) & 
         (df['Mt'] >= 173.0) & (df['Mt'] <= 173.6)]
print(f"\n=== Near experimental point (M_h~125.1, M_t~173.3) ===")
print(f"Count: {len(exp)}")
for _, row in exp.iterrows():
    print(f"  M_t={row['Mt']:.2f}, M_h={row['Mh_calc']:.2f}, Stab={int(row['Stability'])}, S_exact={row['S_exact']:.3f}, S_approx={row['S_approx']:.3f}")
