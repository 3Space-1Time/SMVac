import pandas as pd, numpy as np

df = pd.read_csv('data/numerical_data.csv')
df2 = df.copy()

# -------- 1. Map 2→3→2 across full M_h range 0-135 GeV ----------
def find_pattern_blocks(mh_target):
    tol = 0.02 if mh_target < 100 else 0.005
    chunk = df2[(df2['Mh_calc'] >= mh_target - tol) & 
                (df2['Mh_calc'] <= mh_target + tol)].sort_values('Mt').copy()
    if len(chunk) < 10:
        return None
    stabs = chunk['Stability'].values.astype(int)
    mt_vals = chunk['Mt'].values
    
    blocks = []
    current_s = stabs[0]
    current_start = mt_vals[0]
    for i in range(1, len(stabs)):
        if stabs[i] != current_s:
            blocks.append((int(current_s), current_start, mt_vals[i-1]))
            current_s = stabs[i]
            current_start = mt_vals[i]
    blocks.append((int(current_s), current_start, mt_vals[-1]))
    
    # Find 2→3→2 pattern
    sandwich = None
    for i in range(1, len(blocks)-1):
        if blocks[i-1][0] == 2 and blocks[i][0] == 3 and blocks[i+1][0] == 2:
            sandwich = {
                'lower_2': (blocks[i-1][0], blocks[i-1][1], blocks[i-1][2]),
                'mid_3': (blocks[i][0], blocks[i][1], blocks[i][2]),
                'upper_2': (blocks[i+1][0], blocks[i+1][1], blocks[i+1][2]),
                'sandwich_width': blocks[i][2] - blocks[i][1],
                'lower_width': blocks[i-1][2] - blocks[i-1][1],
                'upper_width': blocks[i+1][2] - blocks[i+1][1]
            }
            break
    return {'Mh': mh_target, 'n_pts': len(chunk), 'blocks': ''.join(str(b[0]) for b in blocks), 'n_blocks': len(blocks), 'sandwich': sandwich}

print("=== FULL M_h SCAN 0-135 GeV ===")
print(f"{'Mh':>6} {'pts':>5} {'blks':>4} {'pattern':>25} {'has_sw':>7} {'sw_w':>6} {'lo2_start':>9} {'lo2_end':>7} {'mid3_s':>7} {'mid3_e':>7} {'up2_s':>7} {'up2_e':>7} {'lo2_w':>6} {'up2_w':>6}")
print('-' * 115)
for mh_target in np.arange(0, 135.25, 0.25):
    r = find_pattern_blocks(mh_target)
    if r is None:
        continue
    sw_str = 'YES' if r['sandwich'] else 'NO'
    sw = r['sandwich']
    if sw:
        print(f"{r['Mh']:>6.2f} {r['n_pts']:>5} {r['n_blocks']:>4} {r['blocks']:>25} {sw_str:>7} "
              f"{sw['sandwich_width']:>6.1f} "
              f"{sw['lower_2'][1]:>7.1f} {sw['lower_2'][2]:>7.1f} "
              f"{sw['mid_3'][1]:>7.1f} {sw['mid_3'][2]:>7.1f} "
              f"{sw['upper_2'][1]:>7.1f} {sw['upper_2'][2]:>7.1f} "
              f"{sw['lower_width']:>6.1f} {sw['upper_width']:>6.1f}")
    else:
        print(f"{r['Mh']:>6.2f} {r['n_pts']:>5} {r['n_blocks']:>4} {r['blocks']:>25} {sw_str:>7}")
