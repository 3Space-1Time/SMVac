import pandas as pd, numpy as np

df = pd.read_csv('data/numerical_data.csv')
computed = df[df['S_exact'] != -1.0].copy()

# Focus on M_h near 125 GeV, zoom in on M_t near experimental value
mh_target = 125.1
chunk = computed[(computed['Mh_calc'] >= mh_target - 0.5) & (computed['Mh_calc'] <= mh_target + 0.5)].sort_values('Mt')
print(f'M_h ~ 125: {len(chunk)} points')
print(f'  M_t range: [{chunk["Mt"].min():.1f}, {chunk["Mt"].max():.1f}]')
print()
for _, row in chunk.iterrows():
    mt = row['Mt']
    mh = row['Mh_calc']
    s = int(row['Stability'])
    sex = row['S_exact']
    print(f'  M_t={mt:.2f}, M_h={mh:.2f}, Stab={s}, S_exact={sex:.3f}')
