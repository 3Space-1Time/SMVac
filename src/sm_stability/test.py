import pandas as pd
df = pd.read_csv('../../results/data_interim.csv')
row = df[(abs(df['Mh'] - 230) < 0.1) & (abs(df['Mt'] - 50) < 0.1)]
print("Original interim data:", row)
