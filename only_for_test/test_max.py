import numpy as np
import pandas as pd


I=[]
a=pd.DataFrame({'a':[1,8,7,1,8],'b':[7,8,6,6,3],'c':[2,9,5,9,7],'d':[8,0,3,8,4]},index=[9,8,7,6,5])
print(a)
print(a.max(axis=0))
[print(i) for i in range(a.shape[1]) if a.loc[:,i].max()==8]
