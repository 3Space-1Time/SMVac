import pandas as pd
d = pd.read_csv('data/diagnostic_chunk_43846.csv')
print(f'diagnostic_chunk_43846.csv: {len(d)} rows')
print(f'  Mt: {d["Mt"].min():.1f} to {d["Mt"].max():.1f}')
print(f'  Mh: {d["Mh_calc"].min():.1f} to {d["Mh_calc"].max():.1f}')
print(f'  Stability counts: {d["Stability"].value_counts().to_dict()}')
print(f'  Computed (S_kin>0): {(d["S_kin"]>0).sum()}')

# Determine remaining: (Mt, Mh) structure: 201 pts per Mt slice
# Last row's Mt
last_mt = d['Mt'].max()
print(f'Last Mt in chunk: {last_mt:.2f}')
# What index should the next chunk start at?
# The solver writes index starting from the input start_idx
print(f'Number of rows written: {len(d)}')
print(f'So next start should be: 43846 + {len(d)} = {43846 + len(d)}')

# Also compute what Mt value this corresponds to
n_mh_per_mt = 201  # Mh 0 to 100 in 0.5 steps
current_mt_index = (last_mt - 90.0) / 0.25
print(f'Current Mt index: {current_mt_index:.1f} (should be ~{(43846+len(d))/n_mh_per_mt:.1f})')
