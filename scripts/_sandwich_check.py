import pandas as pd, numpy as np

df = pd.read_csv('data/numerical_data.csv')

# Focus on M_h = 5 GeV - check if sandwiched pattern exists
for mh_target in [5.0, 5.25, 5.5, 5.75]:
    tol = 0.02
    chunk = df[(df['Mh_calc'] >= mh_target - tol) & 
               (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt')
    if len(chunk) < 5:
        continue
    print(f'=== M_h ~ {mh_target} GeV ({len(chunk)} points) ===')
    stabs = chunk['Stability'].values.astype(int)
    transitions = []
    prev = stabs[0]
    for i, s in enumerate(stabs):
        if s != prev:
            transitions.append((chunk['Mt'].values[i], int(s)))
            prev = s
    tr_str = ', '.join([f'{mt:.0f}->{s}' for mt,s in transitions[:12]])
    print(f'  Stability transitions: {tr_str}')
    neg = chunk[chunk['S_exact'] < 0]
    print(f'  Negative S: {len(neg)} pts')
    sentinel = chunk[chunk['S_exact'] == -1.0]
    print(f'  Sentinel (S=-1): {len(sentinel)} pts')
    stab_counts = chunk['Stability'].value_counts().sort_index()
    print(f'  Stability counts: {stab_counts.to_dict()}')
    
    # Show the first and last few points  
    print(f'  First 5: M_t={chunk["Mt"].values[:5]}, Stab={stabs[:5]}')
    print(f'  Last 5: M_t={chunk["Mt"].values[-5:]}, Stab={stabs[-5:]}')
    # Check if there's a sandwiched pattern: find contiguous blocks
    blocks = []
    current_stab = stabs[0]
    current_start = chunk['Mt'].values[0]
    for i, s in enumerate(stabs):
        if s != current_stab:
            blocks.append((current_stab, current_start, chunk['Mt'].values[i-1]))
            current_stab = s
            current_start = chunk['Mt'].values[i]
    blocks.append((current_stab, current_start, chunk['Mt'].values[-1]))
    block_str = ', '.join([f'{b[0]}:[{b[1]:.1f},{b[2]:.1f}]' for b in blocks])
    print(f'  Contiguous blocks: {block_str}')
    print()
