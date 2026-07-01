import pandas as pd
d = pd.read_csv('data/diagnostic_chunk_50752.csv')
print(f'diagnostic_chunk_50752.csv: {len(d)} rows')
print(f'  Mt: {d["Mt"].min():.1f} to {d["Mt"].max():.1f}')
print(f'  Mh: {d["Mh_calc"].min():.1f} to {d["Mh_calc"].max():.1f}')
print(f'  Stability counts: {d["Stability"].value_counts().to_dict()}')
print(f'  Computed (S_kin>0): {(d["S_kin"]>0).sum()}')
last_mt = d['Mt'].max()
print(f'Last Mt: {last_mt:.2f}')
print(f'Rows written: {len(d)}')
print(f'Next start index: 50752 + {len(d)} = {50752 + len(d)}')
# What M_t does that correspond to?
n_mh = 201
next_mt = 90.0 + ((50752 + len(d)) / n_mh) * 0.25
print(f'Corresponds to Mt ~ {next_mt:.2f}')
