import pandas as pd, numpy as np

d = pd.read_csv('data/diagnostic_chunk_0.csv')
print('Total rows:', len(d))
print('Mt range:', f'{d["Mt"].min():.1f} - {d["Mt"].max():.1f}')
print('Mh range:', f'{d["Mh_calc"].min():.1f} - {d["Mh_calc"].max():.1f}')
stab_counts = d['Stability'].value_counts().sort_index()
print('Stability counts:')
print(stab_counts)

computed = d[d['S_kin'] > 0]
print(f'\nComputed (S_kin > 0): {len(computed)}')

neg = d[(d['S_exact'] < 0) & (d['S_kin'] > 0)]
print(f'Negative S_exact (with S_kin>0): {len(neg)}')
if len(neg) > 0:
    print('\nSample of negative-S points:')
    neg_sorted = neg.sort_values('S_exact')
    cols = ['Mt','Mh_calc','Stability','S_exact','S_kin','S_pot','lambda_R','R_opt']
    print(neg_sorted[cols].head(10).to_string())
    print('...most negative:')
    print(neg_sorted[cols].tail(5).to_string())
    neg['virial'] = neg['S_kin'] / neg['S_pot'].abs()
    print('\nVirial ratio (S_kin/|S_pot|) for negative-S:')
    print(neg['virial'].describe())

pos = d[(d['S_exact'] > 0) & (d['S_kin'] > 0)]
print(f'\nPositive S_exact (with S_kin>0): {len(pos)}')
if len(pos) > 0:
    pos['virial'] = pos['S_kin'] / pos['S_pot']
    print('\nSample of positive-S points:')
    print(pos[['Mt','Mh_calc','Stability','S_exact','S_kin','S_pot','virial']].head(10).to_string())
    print('\nVirial ratio for positive-S:')
    print(pos['virial'].describe())

# Check at Mh=5
for mh_target in [5.0]:
    tol = 0.3
    chunk = d[(d['Mh_calc'] >= mh_target - tol) & (d['Mh_calc'] <= mh_target + tol)].sort_values('Mt')
    print(f'\n--- Mh ~ {mh_target} (N={len(chunk)} pts) ---')
    print(f'{"Mt":>8} {"Stab":>4} {"S_exact":>12} {"S_kin":>12} {"S_pot":>12} {"lam_R":>10} {"R_opt":>10} {"virial":>8}')
    print('-' * 85)
    for _, row in chunk.iterrows():
        if row['S_kin'] > 0:
            vir = row['S_kin'] / abs(row['S_pot']) if row['S_pot'] != 0 else 999
            print(f'{row["Mt"]:>8.1f} {int(row["Stability"]):>4} {row["S_exact"]:>12.3f} {row["S_kin"]:>12.3f} {row["S_pot"]:>12.3f} {row["lambda_R"]:>10.6f} {row["R_opt"]:>10.3f} {vir:>8.3f}')
