import pandas as pd, numpy as np

df = pd.read_csv('data/numerical_data.csv')

# Check near-boundary error at M_h=125
# Look at M_t where S_exact crosses S_threshold ~ 300
mh_target = 125.0
tol = 0.1
chunk = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].copy()
chunk = chunk[chunk['S_exact'] != -1.0].sort_values('Mt')
print(f'M_h~125 GeV boundary check:')
print(f'  M_t range: [{chunk["Mt"].min():.1f}, {chunk["Mt"].max():.1f}]')
for _, row in chunk[::10].iterrows():
    mt = row['Mt']
    sex = row['S_exact']
    sapp = row['S_approx']
    if sex > 0:
        err = abs(sex - sapp) / sex * 100
    else:
        err = float('nan')
    stab = int(row['Stability'])
    print(f'  M_t={mt:.1f}: S_exact={sex:.1f}, S_approx={sapp:.1f}, err={err:.1f}%, Stab={stab}')

print()

# Also check M_h=50 boundary
mh_target = 50.0
chunk2 = df[(df['Mh_calc'] >= mh_target - tol) & (df['Mh_calc'] <= mh_target + tol)].copy()
chunk2 = chunk2[chunk2['S_exact'] != -1.0].sort_values('Mt')
print(f'M_h~50 GeV boundary check:')
for _, row in chunk2[::10].iterrows():
    mt = row['Mt']
    sex = row['S_exact']
    sapp = row['S_approx']
    if sex > 0:
        err = abs(sex - sapp) / sex * 100
    else:
        err = float('nan')
    stab = int(row['Stability'])
    print(f'  M_t={mt:.1f}: S_exact={sex:.1f}, S_approx={sapp:.1f}, err={err:.1f}%, Stab={stab}')
