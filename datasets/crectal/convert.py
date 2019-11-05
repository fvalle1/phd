import pandas as pd
import numpy as np

df = pd.read_csv("crectal.csv", index_col=0)

def idx(x):
    if x.max()==0:
        return np.nan
    else:
        return df.columns[np.argmax(x.values)]

df.insert(0,'CMS',df.apply(idx, axis=1))
print(df.head())
old_index=df.index.values
df.index=[i[:12] for i in old_index]
print(df.head())
df.to_csv("files_CMS.dat")
