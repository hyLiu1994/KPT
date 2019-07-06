import os
import pandas as pd
import pickle
import re
import numpy as np
import datetime
import time

# coding = utf-8
'''
需要给定原始的OJ数据集
原始数据集包含一下两部分
    1.用户的签到记录（xxx_RawSubmitRecord.txt）
        每行为一条提交记录,具体包含一下属性
            提交ID,提交时间,返回状态,题目编号,执行时间,内存占用,代码长度,用户名
            以上每一行不同属性以3个空格分割
    2.知识点题目映射关系（xxx_RawKnowledge2Problem.txt）
        每行为某一知识点的包含的题目，具体形式如下(n表示某个知识点包含题目的数量)
            知识点名:题目ID1,题目ID2,......,题目IDn
        以上每一行具体表示为以下形式:
            knowledgeName:ProblemID1,ProblemID2,ProblemID3,...,ProblemIDn
'''
class OJDataProcessor(object):
    #用户与用户ID之间的转换关系，字典中每个用户都是合法的
    userName2userId = {}
    userId2userName = {}
    #题目和题目ID之间的转换关系，字典中每一道题目都是合法的
    problemName2problemId = {}
    problemId2problemName = {}
    # 知识点和ID之间的转换关系，字典中知识点都是合法的
    knowledgeName2knowledgeId = {}
    knowledgeId2KnowledgeName = {}
    #合法的所有提交记录
    submitRecord = []
    #题目-知识点矩阵
    QMatrix = []
    #时间-用户-题目tensor
    SubmitTensor = []
    # ###定义了一个变量用来保存包含知识点题的id号
    prombleHasKnowledge = []
    #数据集名称
    DataName = ""
    #临时文件夹位置
    TmpDir = ""

    def __init__(self,DataName = 'hdu',TmpDir = 'Data/'):
        self.DataName = DataName
        self.TmpDir = TmpDir
        self.Knowledge2ProblemPath = self.TmpDir + self.DataName + '_RawKnowledge2Problem.txt'
        self.RawSubmitRecordPath = self.TmpDir + self.DataName + '_RawSubmitRecord.txt'
        print('Knowledge2ProblemPath:',self.Knowledge2ProblemPath,'\nRawSubmitRecordPath:', self.RawSubmitRecordPath)
        self.RawSubmitRecord2CSV()

    #将原始提交记录文件转化为CSV文件方便后期处理
    #转化后名称为"xxx_RawSubmitRecord.csv"
    #过程中要加入判断当前文件是否存在，存在不用进行二次处理
    #输出提示是否需要重新计算
    #如果处理的话,加上处理的进度条
    def RawSubmitRecord2CSV(self):
        if os.path.exists(self.TmpDir + self.DataName+'_RawSubmitRecord.csv'):
            str = input("reRawSubmitRecord2CSV y/n：")
            if str =='n':
                return
        print('is RawSubmitRecord2CSV')
        count=0
        thefile=open(self.RawSubmitRecordPath)
        while True:
            buffer=thefile.read(1024*8192)
            if not buffer:
                break
            count+=buffer.count('\n')
        thefile.close()
        w_f = open(self.TmpDir + self.DataName+'_RawSubmitRecord.csv', 'w', encoding='UTF-8')
        num_line = 0
        with open(self.RawSubmitRecordPath, 'r', encoding='utf-8') as f:
            for line in f:
                num_line += 1
                if num_line %1000000 == 0:
                    print('has loading line :',num_line,'/',count)
                v = line.split('   ')
                if len(v) != 10:
                    continue
                v[8]= re.sub(r',+','_',v[8])
                out = v[8].strip()
                for i in range(1, 8):
                    out += ',' + v[i].strip()
                out += '\r'
                w_f.writelines(out)
        print('LoadSubmitRecordData end')


    #将原始知识点题目映射文件转化为CSV文件方便后期处理
    #转化后名称为"xxx_Knowledge2Problem.csv"3
    #过程中要加入判断当前文件是否存在，存在不用进行二次处理
    #输出提示是否需要重新计算
    #如果处理的话,加上处理的进度条
    # ######这个函数需要problemName2problemId字典，所以要放到LoadSubmitRecordData后去执行
    def RawKnowledge2CSV(self):
        if os.path.exists(self.TmpDir + self.DataName+'_RawKnowledge.csv'):
            s = input("reRawKnowledge y/n：")
            if s =='n':
                # 将knowledgeName2knowledgeId 和 knowledgeId2knowledgeName还有prombleHasKnowledge加进来
                f5 = open(self.TmpDir + self.DataName + '_knowledgeName2knowledgeId.txt', 'r', encoding='utf-8')
                f6 = open(self.TmpDir + self.DataName + '_knowledgeId2knowledgeName.txt', 'r', encoding='utf-8')
                f7 = open(self.TmpDir + self.DataName + '_prombleHasKnowledg', 'r', encoding='utf-8')
                self.knowledgeName2knowledgeId = eval(f5.read())
                self.knowledgeId2KnowledgeName = eval(f6.read())
                for row in f7:
                    row = row.split(' ')
                    for i in range(0, len(row) - 1):
                        self.prombleHasKnowledge.append(row[i])
                    self.prombleHasKnowledge.append(row[len(row) - 1].strip('\n'))
                f5.close(), f6.close(), f7.close()
                return
        print('is RawKnowledge2CSV')
        file = open(self.Knowledge2ProblemPath, 'r')
        num = 0
        # KP是知识点-题目名字典
        kp = {}
        # 存知识点的名字
        name = None
        for row in file:
            line = row.split(':')
            name = line[0]
            line = line[1]
            line = line.split(',')
            kp[name] = []
            for i in range(len(line)):
                # 将之前的数据POJ1111，处理成1111
                line[i] = line[i].split(self.DataName)
                kp[name].append(line[i][1])
        file.close()
        file2 = open(self.TmpDir + self.DataName + '_RawKnowledge.csv', 'w', encoding='UTF-8')
        for j in kp.keys():
            self.knowledgeName2knowledgeId[j] = num
            self.knowledgeId2KnowledgeName[num] = j
            num += 1
        for j in kp.keys():
            file2.write(str(self.knowledgeName2knowledgeId[j]))
            file2.write(':')
            k = 0
            w = 0
            for i in kp[j]:
                k += 1
                if int(i) in self.problemName2problemId.keys() and k != len(kp[j]) and w == 0:
                    w = 1
                    file2.write(str(self.problemName2problemId[int(i)]))
                    self.prombleHasKnowledge.append(self.problemName2problemId[int(i)])
                    continue
                if int(i) in self.problemName2problemId.keys() and k != len(kp[j]):
                    file2.write(',')
                    file2.write(str(self.problemName2problemId[int(i)]))
                    self.prombleHasKnowledge.append(self.problemName2problemId[int(i)])
                if k == len(kp[j]):
                    str1 = i.split('\n')[0]
                    if int(str1) in self.problemName2problemId.keys():
                        if w == 1:
                            file2.write(',')
                        file2.write(str(self.problemName2problemId[int(str1)]))
                        self.prombleHasKnowledge.append(self.problemName2problemId[int(i)])
            file2.write('\n')
        # 将字典存到txt文件
        # 读取方法是
        # f = open(self.TmpDir + self.DataName +'_knowledgeName2knowledgeId.txt','r')
        #         a = f.read()
        #         dict_name = eval(a)
        #         print(dict_name)
        #         f.close()
        file3 = open(self.TmpDir + self.DataName +'_knowledgeName2knowledgeId.txt', 'w', encoding='utf-8')
        file4 = open(self.TmpDir + self.DataName +'_knowledgeId2knowledgeName.txt', 'w', encoding='utf-8')
        file3.write(str(self.knowledgeName2knowledgeId))
        file4.write(str(self.knowledgeId2KnowledgeName))
        file2.close()
        file3.close()
        file4.close()
        # 将prombleHasKnowledge这个列表写入文件中
        # 读的方法是
        # f7 = open(self.TmpDir + self.DataName +'_prombleHasKnowledg', 'r')
        # for row in f7:
        #     row = row.split(' ')
        #     for i in range(0, len(row)-1):
        #         self.prombleHasKnowledge.append(row[i])
        #     self.prombleHasKnowledge.append(row[len(row)-1].strip('\n'))
        file5 = open(self.TmpDir + self.DataName +'_prombleHasKnowledge', 'w', encoding='utf-8')
        s = str(self.prombleHasKnowledge).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file5.write(s)
        file5.close()
        print('LoadKnowledgeData end')

    #userLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
    #ProblemLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
    #timeLC = [起始时间（单位秒），终止时间（秒）]
    #LC2Str将限制条件映射为一个字符串
    def LC2Str(self,userLC,problemLC,timeLC,OnlyRight):
        c = re.sub(r':', '.', str(timeLC))
        return ('userLC_'+str(userLC)+'_problemLC_'+str(problemLC)+'_timeLC_'+c+'_OnlyRight_'+str(OnlyRight))

    #userLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
    #ProblemLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
    #timeLC = [起始时间（单位秒），终止时间（秒）]
    # 当OnlyRight为真的时候，只考虑Accepted，其它所有情况划分为一类，等OnlyRight为假的时候
    # 分为这几种情况dic_status = {'Accepted': 1, 'Wrong Answer': 0, 'Time Limit Exceeded': 2, 'other': 3}
    #根据限制条件过滤数据
    #最终要获得userName-userId,userId-userName,problemRawId-problemId,problemId-ProblemRawId四个字典，要保证存在在字典中的每一个题目与用户都是满足限制条件的，以及满足限制条件的提交记录。
    #加进度条并保持求的5个结果,最终结果保持在TmpDir路径下
    #五个文件的前缀应该是DataName+LC2Str(userLC,problemLC,timeLC),五个文件各自的后缀名称你可以自行设计
    def FilterSubmitRecordData(self,userLC,problemLC,timeLC,OnlyRight):
        print('is FilterSubmitRecordData')
        dic_status = {'Accepted': 1, 'Wrong Answer': 0, 'Time Limit Exceeded': 2, 'other': 3}
        def convert_statue(value):
            value = value.strip()
            if OnlyRight:
                if value == 'Accepted':
                    return 1
                else:
                    return 0
            else:
                if value in dic_status.keys():
                    return dic_status[value]
                else:
                    return dic_status['other']

        def delet_pro(df,filter_id,filter_condition):
            def cal_pro(v):
                total_num = v[filter_id].count()
                if filter_id == 'name':
                    ac_num = v[v['status'] == 1]['PID'].nunique()
                else:
                    ac_num = v[v['status'] == 1]['PID'].count()
                v['total'] = total_num
                v['ac_num'] = ac_num
                v['pro'] = ac_num / total_num
                return v[[filter_id, 'total', 'ac_num', 'pro']]

            judge = df.groupby(filter_id).apply(cal_pro)
            judge.drop_duplicates(subset=[filter_id], inplace=True)
            judge = judge[(judge['pro'] <filter_condition[2]) | (judge['pro'] >filter_condition[3])|(judge['total'] < filter_condition[0]) |(judge['total'] >filter_condition[1])]
            print(judge)
            return judge


        df = pd.read_csv(self.TmpDir + self.DataName+'_RawSubmitRecord.csv', delimiter=',', lineterminator='\r', header=None, index_col=False)
        df.columns = ['name', 'sub_time', 'status', 'PID', 'op_time', 'memeory', 'codeSize', 'lan']
        #filter time
        df = df[(df['sub_time']>=timeLC[0]) & (df['sub_time']<=timeLC[1])]
        #status
        df['status'] = df['status'].apply(convert_statue)
        #filter problem return need delete problem
        print('is filtering problem\nthe filtered problem list is ')
        judge_pro = delet_pro(df,'PID',problemLC)
        df = df[-df['PID'].isin(list(judge_pro['PID']))]
        #filter student return need delete student
        print('is filtering student \nthe filtered student list is')
        judge_stu = delet_pro(df,'name',userLC)
        df = df[-df['name'].isin(list(judge_stu['name']))]
        #filter end
        print('is writing to file')
        t = self.LC2Str(userLC,problemLC,timeLC,OnlyRight)
        print(t)
        w_f1 = open(self.TmpDir + self.DataName+
               t+ '_userName2userId.pkl', 'wb' )
        w_f2 = open(self.TmpDir + self.DataName+
                t+ '_userId2userName.pkl', 'wb')
        w_f3 = open(self.TmpDir + self.DataName+
               t+ '_problemName2problemId.pkl', 'wb' )
        w_f4 = open(self.TmpDir + self.DataName+
                t+ '_problemId2problemName.pkl', 'wb')

        def tonameid(name):
            if name not in self.userName2userId.keys():
                id = len(self.userName2userId)
                self.userName2userId[name] = id
                self.userId2userName[id] = name
            return self.userName2userId[name]
        def toproid(name):
            if name not in self.problemName2problemId.keys():
                id = len(self.problemName2problemId)
                self.problemName2problemId[name] = id
                self.problemId2problemName[id] = name
            return self.problemName2problemId[name]
        df['name'] =df['name'].apply(tonameid)
        df['PID'] = df['PID'].apply(toproid)
        df = df.sort_values(by=['name', 'sub_time', 'PID'])
        print(df)
        df[['name','PID','status']].to_csv(self.TmpDir + 'hdu_EERNN_input.csv', index=False, header=False, encoding='utf-8')
        df[['sub_time','name','PID','status']].to_csv(self.TmpDir + 'hdu_KPT.csv', index=False, header=False, encoding='utf-8')
        pickle.dump(self.userName2userId, w_f1, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.userId2userName, w_f2, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.problemName2problemId, w_f3, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.problemId2problemName, w_f4, pickle.HIGHEST_PROTOCOL)
        w_f1.close(),w_f2.close(),w_f3.close(),w_f4.close()



    #导入满足限制条件的数据，如果之前没有处理过直接调用FilterSubmitRecordData函数,如果处理过直接读取之前结果，构建6个dict以及合法的提交记录list。
    def LoadSubmitRecordData(self,userLC,problemLC,timeLC,OnlyRight):
        if os.path.exists(self.TmpDir + self.DataName+'_EERNN_input.csv'):
            str = input("ReLoadSubmitRecordData (if you change userLC,problemLC,timeLC,OnlyRight.or reRawSubmitRecord2CSV ,please input y) y/n：");
            if str == 'y':
                self.FilterSubmitRecordData(userLC, problemLC, timeLC,OnlyRight)
        else:
            self.FilterSubmitRecordData(userLC, problemLC, timeLC,OnlyRight)
        print('is LoadSubmitRecordData')
        with open(self.TmpDir + self.DataName+'_EERNN_input.csv','r') as f:
            for line in f:
                fields = line.strip().split(',')
                student, problem, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
                while (student >= len(self.submitRecord)):
                    self.submitRecord.append([])
                    self.submitRecord[student].append([problem, is_correct])
            print('is open '+ self.TmpDir +self.DataName+' _EERNN_input.csv')
        f1 = open(self.TmpDir + self.DataName+self.LC2Str(userLC,problemLC,timeLC,OnlyRight)+ '_userName2userId.pkl', 'rb')
        f2 = open(self.TmpDir + self.DataName+self.LC2Str(userLC,problemLC,timeLC,OnlyRight)+ '_userId2userName.pkl', 'rb')
        f3 = open(self.TmpDir + self.DataName+self.LC2Str(userLC,problemLC,timeLC,OnlyRight)+ '_problemName2problemId.pkl', 'rb')
        f4 = open(self.TmpDir + self.DataName+self.LC2Str(userLC,problemLC,timeLC,OnlyRight)+ '_problemId2problemName.pkl', 'rb')
        self.userName2userId = pickle.load(f1)
        self.userId2userName = pickle.load(f2)
        self.problemName2problemId = pickle.load(f3)
        self.problemId2problemName = pickle.load(f4)
        f1.close(), f2.close(), f3.close(), f4.close()


    #先判断之前是否计算过，没有的话根据4个dict以及合法的提交List构建时间-用户-提交Tensor以及QMatrix，构建后将构建结果存储在TmpDir里，每个文件名前缀为DataName+LC2Str(userLC,problemLC,timeLC),两个文件后缀自己取。
    #不要忘记加进度条
    def loadQMatrix(self):
        if os.path.exists(self.TmpDir + self.DataName+'_QMatrix.csv'):
            s = input("reQMatix y/n：")
            if s =='n':
                return
        print('is loadSumitAndQMatix')
        # k的key是知识点id ，k的value是知识点对应题目id
        k = {}
        self.RawKnowledge2CSV()
        file = open(self.TmpDir + self.DataName + '_RawKnowledge.csv', 'r')
        for row in file:
            kid = row.split(':')[0]
            Hid = row.split(':')[1]
            k[kid] = []
            hid = Hid.split(',')
            for i in range(len(hid)-1):
                k[kid].append(hid[i])
            k[kid].append(hid[len(hid)-1].strip('\n'))
        self.QMatrix = np.zeros((len(self.knowledgeId2KnowledgeName), len(self.problemName2problemId)))
        print(self.QMatrix.shape)
        for i in k.keys():
            for j in k[i]:
                row = int(i)
                line = int(j)
                self.QMatrix[row][line] = 1
        Q_Matrix = pd.DataFrame(self.QMatrix)
        Q_Matrix.to_csv(self.TmpDir + self.DataName+'_QMatrix.csv')
        # 如果你想在csv文件中读出Q矩阵，用下面的代码
        # Q_matrix = pd.read_csv(self.TmpDir + self.DataName+'_QMatrix.csv').as_matrix()[
        #            0:self.QMatrix.shape[0]+1, 1:self.QMatrix.shape[1]+2]
        # print(Q_matrix)

    # 生成最后kpt要输入的数据，第一列是时间窗口的id，第二列是学生id，第三列是题目id，第四列是正确与否
    def loadSubmitRecordOfTimeWindows(self, timeLC, timedivid):
        # 先将时间转换成秒
        time1 = timeLC[0]
        time2 = timeLC[1]
        # 将'2018-01-01 23:47:31'的形式转换为秒的形式
        def timetosecond(time1):
            t11 = time1.split(' ')[0].split('-')
            if int(t11[1]) / 10 == 0:
                t11[1] = int(t11[1]) - 10
            if int(t11[1]) / 10 == 0:
                t11[2] = int(t11[1]) - 10
            t12 = time1.split(' ')[1].split(':')
            t1 = datetime.datetime(int(t11[0]), int(t11[1]), int(t11[2]), int(t12[0]), int(t12[1]), int(t12[2]))
            t1 = time.mktime(t1.timetuple())
            return t1
        tbegin = timetosecond(time1)
        tend = timetosecond(time2)
        timeinterval = (tend-tbegin)/(timedivid+1)
        file = open(self.TmpDir + 'hdu_KPT.csv', 'r', encoding='utf-8')
        file2 = open(self.TmpDir + 'hdu_KPT_input.csv', 'w', encoding='utf-8')
        for line in file:
            line = line.split(',')
            if len(line) != 0:
                sub_time = line[0]
                Uid = line[1]
                PID = line[2]
                status = line[3]
                t = timetosecond(sub_time)
                tw = (t - tbegin) // timeinterval
                if tw >= timedivid:
                    tw = timedivid-1
                sub_time = tw
                if int(line[2]) in self.prombleHasKnowledge:
                    print('wewewewewe')
                    file2.write(str(int(sub_time)) + ','+ str(Uid)+','+str(PID)+','+str(status))
        file.close()
        file2.close()




tmp = OJDataProcessor()
tmp.LoadSubmitRecordData([15,1000000,0.06,1],[10,1000000,0.02,1],['2018-11-22 20:45:48','2018-11-29 11:22:08'],OnlyRight=True)
tmp.loadQMatrix()
tmp.loadSubmitRecordOfTimeWindowsubmitRecord(['2018-11-22 20:45:48','2018-11-29 11:22:08'], 12)
print(tmp.submitRecord)
print(tmp.userName2userId)
print(tmp.problemName2problemId)
print(tmp.prombleHasKnowledge)
print("END")
