import pandas as pd, numpy as np

df = pd.read_csv('data/numerical_data.csv')

def find_sandwich_blocks(stabs, mt_vals):
    blocks = []
    current_s = stabs[0]
    current_start = mt_vals[0]
    for i, s in enumerate(stabs):
        if s != current_s:
            blocks.append((int(current_s), current_start, mt_vals[i-1]))
            current_s = s
            current_start = mt_vals[i]
    blocks.append((int(current_s), current_start, mt_vals[-1]))
    return blocks

results = []
for mh_target in np.arange(0, 100, 0.25):
    tol = 0.02
    chunk = df[(df['Mh_calc'] >= mh_target - tol) & 
               (df['Mh_calc'] <= mh_target + tol)].sort_values('Mt')
    if len(chunk) < 10:
        continue
    stabs = chunk['Stability'].values.astype(int)
    mt_vals = chunk['Mt'].values
    blocks = find_sandwich_blocks(stabs, mt_vals)
    
    has_sandwich = False
    for i in range(1, len(blocks)-1):
        if blocks[i-1][0] == 2 and blocks[i][0] == 3 and blocks[i+1][0] == 2:
            has_sandwich = True
            sandwich_mt_start = blocks[i][1]
            sandwich_mt_end = blocks[i][2]
            sandwich_width = sandwich_mt_end - sandwich_mt_start
            lower_flank = (blocks[i-1][1], blocks[i-1][2])
            upper_flank = (blocks[i+1][1], blocks[i+1][2])
    # Also look for 2->3->oscillations->2 pattern
    # Count sequential 3's between 2's
    in_region = False
    region_starts = []
    region_ends = []
    for b in blocks:
        if b[0] == 2 and not in_region:
            pass
        elif b[0] == 3 and not in_region:
            in_region = True
            region_starts.append(b[1])
        elif b[0] == 2 and in_region:
            in_region = False
            region_ends.append(b[1])
    
    # Check for S_exact < 0
    neg_in = chunk[chunk['S_exact'] < 0]
    sentinel = chunk[chunk['S_exact'] == -1.0]
    
    results.append({
        'Mh': mh_target,
        'n_pts': len(chunk),
        'n_blocks': len(blocks),
        'pattern': ''.join(str(b[0]) for b in blocks),
        'has_sandwich': has_sandwich,
        'sandwich_width': sandwich_width if has_sandwich else 0,
        'sandwich_mt_start': sandwich_mt_start if has_sandwich else 0,
        'sandwich_mt_end': sandwich_mt_end if has_sandwich else 0,
        'lower_flank_start': lower_flank[0] if has_sandwich else 0,
        'lower_flank_end': lower_flank[1] if has_sandwich else 0,
        'upper_flank_start': upper_flank[0] if has_sandwich else 0,
        'upper_flank_end': upper_flank[1] if has_sandwich else 0,
        'n_neg_S': len(neg_in),
        'n_sentinel': len(sentinel)
    })

print(f"{'Mh':>6} {'pts':>5} {'blks':>4} {'sandwich':>9} {'Mt_start':>8} {'Mt_end':>7} {'width':>6} {'low_Mt':>7} {'up_Mt':>7} {'neg_S':>6} {'sent':>5}")
print('-' * 80)
for r in results:
    if r['has_sandwich']:
        print(f"{r['Mh']:>6.2f} {r['n_pts']:>5} {r['n_blocks']:>4} {'YES':>9} "
              f"{r['sandwich_mt_start']:>8.1f} {r['sandwich_mt_end']:>7.1f} {r['sandwich_width']:>6.1f} "
              f"[{r['lower_flank_start']:.0f},{r['lower_flank_end']:.0f}] "
              f"[{r['upper_flank_start']:.0f},{r['upper_flank_end']:.0f}] "
              f"{r['n_neg_S']:>6} {r['n_sentinel']:>5}")
