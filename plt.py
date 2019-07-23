import matplotlib.pyplot as plt
import numpy as np

KPT_Result=[0.3424973703227374, 0.3037812412903854, 0.3015666808164247, 0.2665089268415772, 0.3122982058745505, 0.31807628579928304, 0.4606175138508871]
Index_Of_KPT_Result=[]
for index,item in enumerate(KPT_Result):
    Index_Of_KPT_Result.append(index)
plt.plot(Index_Of_KPT_Result,KPT_Result)
plt.show()