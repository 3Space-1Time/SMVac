import pandas as pd
d = pd.read_csv('data/diagnostic_chunk_35000.csv')
print(f'diagnostic_chunk_35000.csv: {len(d)} rows')
print(f'  Mt: {d["Mt"].min():.1f} to {d["Mt"].max():.1f}')
print(f'  Mh: {d["Mh_calc"].min():.1f} to {d["Mh_calc"].max():.1f}')
print(f'  Stability counts: {d["Stability"].value_counts().to_dict()}')
print(f'  Computed (S_kin>0): {(d["S_kin"]>0).sum()}')
# The last index
last_idx = d.index[-1]
print(f'  Index range: {d.index[0]} to {last_idx}')
