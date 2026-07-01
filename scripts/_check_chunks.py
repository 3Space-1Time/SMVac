import pandas as pd
for f in ['diagnostic_chunk_0.csv', 'diagnostic_chunk_20000.csv']:
    d = pd.read_csv(f'results/{f}')
    print(f'{f}: {len(d)} rows')
    print(f'  Mt: {d["Mt"].min():.1f} to {d["Mt"].max():.1f}')
    print(f'  Mh: {d["Mh_calc"].min():.1f} to {d["Mh_calc"].max():.1f}')
    print(f'  Stability counts: {d["Stability"].value_counts().to_dict()}')
    print(f'  Computed (S_kin>0): {(d["S_kin"]>0).sum()}')
