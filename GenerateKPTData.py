#coding=utf_8
import numpy as np
import pandas as pd
from OJDataProcessor import *
from Constant import *

'''
整个函数的主要作用：生成KPT_input文.csv文件和Q_Matrixs.csv文件
FilterSubmitRecordDataOf___()的文件输出是：
    _userName2userId.pkl            #用户名字和用户ID的映射
    _userId2userName.pkl            #用户ID和用户名字的映射
    _problemName2problemId.pkl      #问题名字和问题ID的映射
    _problemId2problemName.pkl      #问题ID和问题名字的映射
LoadSubmitRecordOfTimeWindows()的文件输出是：
    _KPT_input.csv                  #生成KPT模型的数据输入文件
LoadQMatrix（）的文件输出是：
    _Q_Matrix.csv                   #生成Q矩阵
'''

def generateKPTData():
    tmp = OJDataProcessor(DataNam,TmDir)
    if(DataNam=='NEULC'):
        tmp.FilterSubmitRecordDataOfNEULC(userLC, problemLC, timeLC_for_KPT, True)
    if(DataNam=='hdu'):
        tmp.FilterSubmitRecordDataOfhdu(userLC, problemLC, timeLC_for_KPT, True)
    tmp.LoadSubmitRecordOfTimeWindows(timeLC,timdivid)
    tmp.LoadQMatrix()
    tmp.DeleteNoUsed()



