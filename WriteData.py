#coding=utf_8
import numpy as np
import pandas as pd
from Constant import *
import re

c=re.sub(r':','.',str(timeLC))
I = [0,3,6,11,12,17,26,29,31,32,37,42,53,75,84,85,97,99,102,103,104,106,109,113,117,157,199]
def writeRMatrix():
    # 待定
    print('is Write R_matrix.dat')
    df = pd.read_csv(TmDir + DataNam +'_'+ c+'_'+str(timdivid)+'_KPT_input.csv', index_col=None, header=None)
    R_matrix = np.zeros((int(df.loc[:, 1].max()) + 1, int(df.loc[:, 2].max()) + 1), dtype=int)
    # 注意这个类型超级重要，不加dtype的话默认是浮点型
    df2 = df[(df[3] == 1) & (df[0] != 11)]
    for index, row in df2.iterrows():
        R_matrix[row[1]][row[2]] = 1
    R_matrix=R_matrix[:,I]
    np.savetxt(TmDir + DataNam + "_" + c + "_" + 'pre11' + '_'+'0.2'+'_'+ "R_matrix.dat", R_matrix, fmt='%d')
    print('write Rmatrix completly!')


def writeQMatrix():
    print('is Write Q_matrix.dat')
    Q_matrix = pd.read_csv(TmDir + DataNam + '_Q_Matrix.csv', index_col=0, header=0).as_matrix()
    Q_matrix=Q_matrix[:,I]
    print(Q_matrix)
    np.savetxt(TmDir + DataNam + '_' +'0.2'+'_'+'Q_Matrix.dat', Q_matrix, fmt='%d')
    print('write Qmatrix completly!')


def writeIMatrix():
    print('is Write I_matrix.dat')
    df = pd.read_csv(TmDir + DataNam +'_'+ c+'_'+str(timdivid)+'_KPT_input.csv', index_col=None, header=None)
    I_matrix = np.zeros((int(df.loc[:, 1].max()) + 1, int(df.loc[:, 2].max()) + 1),dtype=int)
    for index, row in df.iterrows():
        I_matrix[int(row[1])][int(row[2])] = 1
    for i in range(len(I_matrix[0])):
        if np.sum(I_matrix[:, i])/len(I_matrix[0])>0.2:
            print(i)
    I_matrix=I_matrix[:,I]
    print(I_matrix)
    # np.savetxt(TmDir + DataNam + '_' + c +'_' +str(timdivid) + '_' + 'I_matrix.dat', I_matrix, fmt='%d')
    # print('write Imatrix completly!')

# writeRMatrix()
# writeIMatrix()
# writeQMatrix()