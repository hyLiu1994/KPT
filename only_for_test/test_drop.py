import pandas as pd

df = pd.DataFrame([11, 12, 13, 14, 15, 16, 17], index=[0, 1, 2, '2018-01-01', '2018-01-02', 'a', 'b'], columns=['V'])



df.drop([1, 2], inplace=True)
# [df.drop(index,inplace=True) for index in df.iterrows() if index in [1,2]]

I=[13,15]
II=df[df['V'].isin(I)].index
print(II)
df.drop(II,inplace=True)
print(df)