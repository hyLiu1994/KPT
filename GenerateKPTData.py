#coding=utf_8
import numpy as np
import pandas as pd
from OJDataProcessor import *
from Constant import *

# 生成KPT_input文.csv文件和Q_Matrixs.csv文件
#整个项目的文件输入：
#整个项目的文件输出：
def generateKPTData():
    tmp = OJDataProcessor(DataNam,TmDir)
    if(DataNam=='NEULC'):
        tmp.FilterSubmitRecordDataOfNEULC(userLC, problemLC, timeLC_for_KPT, True)
    if(DataNam=='hdu'):
        tmp.FilterSubmitRecordDataOfhdu(userLC, problemLC, timeLC_for_KPT, True)
    tmp.LoadSubmitRecordOfTimeWindows(timeLC,timdivid)
    tmp.LoadQMatrix()
    tmp.DeleteNoUsed()



