import pandas as pd
import sys

if len(sys.argv) < 2:
    exit()

df = pd.read_csv(sys.argv[1] + '.csv').dropna()
df.to_csv(sys.argv[1] + '_nona.csv', index=False)
