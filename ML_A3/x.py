import numpy as np
import pandas as pd

df = pd.read_csv('digits_training.tra', sep=",", skiprows=0)

df=np.array(df)
print(df.shape,df.dtype)
dat=df[:,0:64]
tar1=df[:,64]
X=dat
y=tar1
y1=y[4:65:10]
y1.astype(int)

print(y,y1)