import pandas as pd, numpy as np

df = pd.read_csv('data/numerical_data.csv')
computed = df[df['S_exact'] != -1.0].copy()

for mh_target in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125, 130]:
    tol = 0.3
    chunk = computed[(computed['Mh_calc'] >= mh_target - tol) & 
                     (computed['Mh_calc'] <= mh_target + tol)].sort_values('Mt')
    if len(chunk) < 10:
        continue
    stabs = chunk['Stability'].values.astype(int)
    transitions = []
    prev = stabs[0]
    for i, s in enumerate(stabs):
        if s != prev:
            transitions.append((i, chunk['Mt'].values[i], int(s)))
            prev = s
    neg_in = chunk[chunk['S_exact'] < 0]
    tstr = '...'
    if len(transitions) > 0:
        tstr = ' '.join([f'{mt:.0f}:{s}' for i,mt,s in transitions[:8]])
    nrange = 'none'
    if len(neg_in) > 0:
        nrange = f'[{neg_in["S_exact"].min():.1f},{neg_in["S_exact"].max():.1f}]'
    print(f'M_h~{mh_target:3d}: M_t=[{chunk["Mt"].min():.1f},{chunk["Mt"].max():.1f}]  '
          f'transitions=[{tstr}]  neg_S={len(neg_in):3d}  Sneg={nrange}')
