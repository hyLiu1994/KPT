# coding = utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from generateFigure import create_gif
from OJDataProcessor import *
from GenerateKPTData import *
from WriteData import *

tf.enable_eager_execution()

class KPT(object):
    #loss函数超参数
    lambdap=0.1
    lambdau=0.05
    lambdav=0.05

    # 路径变量
    TmpDir =""
    DataName=""

    # 限制条件
    c=""
    d=""

    #与学习率和遗忘率相关的参数
    delta_t=1
    S=5
    D=2
    r=4

    alpha=0.5

    #学生数量
    num_user = 0
    #题目数量
    num_item = 0
    #知识点数量
    K = 0
    #用户知识熟练度张量[T时刻，userId，knowledgeId]
    U = 0
    #题目知识包含矩阵
    V = 0
    #常量矩阵
    #学生做题矩阵
    I = 0
    #偏序关系矩阵
    I_matrix = 0

    # （标注知识点的题目数量比较少，增加了一个题目id和题目位置的映射，方便以后使用）
    #Q矩阵中题目id和题目位置的映射关系
    QId2QPos={}
    #Q矩阵中题目位置和题目id的映射关系
    QPos2QId={}
    #R矩阵中题目位置和题目id的映射关系
    RPos2RId={}
    #R矩阵中题目id和题目位置的映射关系
    RId2RPos={}

    # 均方根误差
    rmse=[]


 #读取训练数据，与OJDataProcessor进行交互
    def loadData(self):
        # 读取Q矩阵
        Q_matrixs = self.readQmatrix()
        print ("loadQMatrix!")
        # 读取R矩阵
        R = self.readRmatrix()
        print ("loadRMatrix!")
        #读取I矩阵
        I=self.readImatrix()
        print ("loadIMatrix!")
        return [R,I,Q_matrixs]


        # 读取Q矩阵
    def readQmatrix(self):
        df_Q = pd.read_csv(self.TmpDir + self.DataName + '_Q_Matrix.csv', index_col=0, header=0)
        col_val=df_Q.columns.values.tolist()
        for index,item in enumerate(col_val):
            self.QId2QPos[index]=int(item)
        self.QPos2QId = dict(zip(self.QId2QPos.values(), self.QId2QPos.keys()))
        Q_matrix=pd.read_csv(self.TmpDir + self.DataName + '_Q_Matrix.csv', index_col=0, header=0).values
        Q_matrix = Q_matrix.T
        self.num_item,self.K=Q_matrix.shape[0],Q_matrix.shape[1]
        print('Q矩阵', Q_matrix.shape)
        print('read Qmatrix completly!')
        return Q_matrix

        # 读取R矩阵
    def readRmatrix(self):
        df_R = pd.read_csv(self.TmpDir + self.DataName + '_' + self.c + '_' + self.d + '_KPT_input.csv',
                           index_col=None, header=None)
        for index,item in enumerate(df_R[1].unique().tolist()):
            self.RId2RPos[index]=item
        self.RPos2RId = dict(zip(self.RId2RPos.values(), self.RId2RPos.keys()))
        R_matrix = np.zeros((len(df_R.loc[:,0].unique()), len(self.RId2RPos), self.num_item), dtype=int)
        df2_R = df_R[df_R[3] == 1]
        print('R矩阵', R_matrix.shape)
        for index, row in df2_R.iterrows():
            row1 = self.RPos2RId[row[1]]
            row2 = self.QPos2QId[row[2]]
            R_matrix[row[0]][row1][row2] = 1
        self.num_user = len(R_matrix[0])
        self.T = len(R_matrix)
        print('read Rmatrix completly!')
        return R_matrix

        # 读取I矩阵
    def readImatrix(self):
        df_I = pd.read_csv(self.TmpDir + self.DataName + '_' + self.c + '_' + self.d + '_KPT_input.csv',
                           index_col=None, header=None)
        I_matrix = np.zeros((len(df_I.loc[:,0].unique()), len(self.RId2RPos), self.num_item), dtype=int)
        for index, row in df_I.iterrows():
            row1 = self.RPos2RId[row[1]]
            row2 = self.QPos2QId[row[2]]
            I_matrix[row[0]][row1][row2] = 1
        print('I矩阵', I_matrix.shape)
        print('read Imatrix completly!')
        return I_matrix



    def __init__(self,DataName,TmpDir,timeLC,timedivided):
        self.DataName = DataName
        self.TmpDir = TmpDir
        self.timeLC=timeLC
        generateKPTData()
        self.c=re.sub(r':','.',str(self.timeLC))
        self.c=re.sub(r"'","",self.c)
        self.d=str(timedivided)
        self.R,self.I,self.Q_matrixs = self.loadData()

        # 创建随机U矩阵 V矩阵
        self.U = tf.Variable(tf.random_normal([self.T, self.num_user, self.K], stddev=0.35, dtype=tf.float64))
        self.V = tf.Variable(tf.random_normal([self.num_item, self.K], stddev=0.35, dtype=tf.float64))
        self.alpha = tf.Variable(tf.random_normal([self.num_user, 1], stddev=0.35, dtype=tf.float64))
        print("Ramdomly generate Q matrix!")
        K = self.K

        # 创建系数矩阵tmp矩阵
        # Q_matrixs [itemNum, K]
        # t1 [K, K*(K-1)]
        t1 = np.zeros((K, K * (K - 1)), dtype=float)
        for i in range(K):
            for j in range(K - 1):
                t1[i][i * (K - 1) + j] = 1;
                t1[j][i * (K - 1) + j] = -1;
        # 创建偏序关系I矩阵
        self.tmp_all = tf.constant(t1)
        I_matrix = np.dot(self.Q_matrixs, t1)
        I_matrix = I_matrix > 0
        I_matrix = I_matrix.astype('float64')
        self.I_matrix = tf.constant(I_matrix)
        print("Partial order relationship I_matrix generated!")
        print("End of initialization!")

    def fit(self, epochs, learnRate=1e-3):
        print("start fitting")

        # 创建loss函数,这里创建了两个：1是涉及到时间因素的部分 2是涉及到Q先验的部分

        def loss_function1(Uba, t):
            if (Uba == 0):
                los = 1 / 2 * tf.reduce_sum(self.I * (self.R[t] - tf.matmul(self.U[t], self.V, transpose_b=True)) ** 2)
            else:
                los = 1 / 2 * tf.reduce_sum(self.I * (self.R[t] - tf.matmul(self.U[t], self.V,
                                                                            transpose_b=True)) ** 2 + self.lambdau * tf.reduce_sum(
                    (Uba - self.U[t]) ** 2))
            return los

        def loss_function2():
            los = -self.lambdap * tf.reduce_sum(self.I_matrix * tf.math.log(
                tf.sigmoid(tf.matmul(self.V, self.tmp_all)))) + self.lambdav / 2 * tf.reduce_sum(self.V ** 2)
            return los

        def frequen_know():
            for l in open(self.TmpDir + self.DataName + '_' + self.c + '_' + self.d + "_KPT_input.csv", "r", encoding="UTF-8"):
                ls = l.strip().split(',')
                j = int(ls[0])
                id = self.QPos2QId[int(ls[2])]
                for i in range(len(self.Q_matrixs[0])):
                    if self.Q_matrixs[id][i] == 1:
                        f_k[j][i] += 1
            return f_k

        def learn(Ut_1, t, f_k):
            l = Ut_1 * (self.D * f_k[t]) / (f_k[t] + self.r)
            return l

        def forgot(Ut_1):
            f = Ut_1 * tf.to_double(tf.exp(-(self.delta_t) / self.S), name="ToDouble")
            return f

        # 优化器
        optimizer = tf.train.AdamOptimizer(learnRate)

        # 梯度计算
        ##f_k是在时间段t中，知识点k出现的频率,shape=[T*k]
        f_k = np.zeros((self.T, self.K))
        f_k = frequen_know()

        for epoch in range(epochs):
            loss = 0
            start = time.time()
            # 12个时间点
            with tf.GradientTape() as tape:
                loss = loss_function2()
                for t in range(self.T):
                    if t == 0:
                        Uba = 0
                    else:
                        Uba = self.alpha * learn(self.U[t - 1], t, f_k) + (1 - self.alpha) * forgot(self.U[t - 1])
                    loss += loss_function1(Uba, t)

            gradients = tape.gradient(loss, [self.U, self.V, self.alpha])
            optimizer.apply_gradients(zip(gradients, [self.U, self.V, self.alpha]))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss.numpy() / int(self.d)))
            print('Time taken for one epoch {} sec\n'.format(time.time() - start))
        print("fit completly")

    def saveModel(self):
        print('start save model')
        path = self.TmpDir + self.DataName + "_" + self.c + "_" + self.d + '_KPT_Model/KPT.npy'
        U = self.U.numpy()
        V = self.V.numpy()
        alpha = self.alpha.numpy()
        path_limit =self.TmpDir+ self.DataName+'_'+self.c+'_'+self.d+ '_KPT_Model'
        if not os.path.exists(path_limit):
            os.makedirs(path_limit)
        np.save(path, np.array([U, V, alpha]))
        print("KPT model saved!")

    def loadModel(self):
        print("start load latest KPT Model")
        path = self.TmpDir + self.DataName + "_" + self.c + "_" + self.d + '_KPT_Model/KPT.npy'
        [U, V, alpha] = np.load(path)
        self.U = tf.Variable(U)
        self.V = tf.Variable(V)
        self.alpha = tf.Variable(alpha)
        print("load latest KPT Model")


    def Experimental(self):
        def RMSE(R, _R):
            ans = tf.sqrt(tf.reduce_sum((_R - R) ** 2) / (self.num_user * self.num_item))
            return ans

        def MAE(R, _R):
            ans = tf.reduce_sum(tf.abs(_R - R)) / (self.num_user * self.num_item)
            return ans

        RMSE_list = []
        for i in range(self.T - 1):
            RMSE_val = RMSE(tf.matmul(self.U[i], self.V, transpose_b=True), self.R[i + 1])
            RMSE_list.append(RMSE_val.numpy())
            print("time windows Id:", i + 2, "  RMSE:", RMSE_list[-1])

        MAE_list = []
        for i in range(self.T - 1):
            MAE_val = MAE(tf.matmul(self.U[i], self.V, transpose_b=True), self.R[i + 1])
            MAE_list.append(MAE_val.numpy())
            print("time windows Id:", i + 2, "  MAE:", MAE_list[-1])

        return RMSE_list, MAE_list

    def user_knowledge_plt(self, timeList, userId, saveFileName = "./test.png"):
        #数据
        file = open('./'+self.TmpDir+'Knowledge_Problem/'+self.DataName+'_knowledgeName2knowledgeId.txt', 'r',
                    encoding='utf-8')
        file = eval(file.read())
        knowledgeLabel = list(file.keys())
        maxKnowledge = 10
        data = []
        for i in range(len(timeList)):
            U = self.U.numpy()[timeList[i]]
            U = (U - U.min(axis = 0))/ (U.max(axis = 0) - U.min(axis =0))
            U = U[userId] * maxKnowledge
            data.append(U)
        #标签
        labels = np.array(knowledgeLabel)
        #数据个数
        dataLenth = len(data[0])

        angles = np.linspace(0, 2*np.pi, dataLenth, endpoint=False)
        for i in range(len(data)):
            data[i] = np.concatenate((data[i], [data[i][0]])) # 闭合
        angles = np.concatenate((angles, [angles[0]])) # 闭合

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)# polar参数！！
        #ax.plot(angles, data, 'bo-', linewidth=2)# 画线
        #ax.fill(angles, data, facecolor='r', alpha=0.25)# 填充
        for i in range(len(data)):
            if (len(data)>1):
                ax.plot(angles, data[i], linewidth=2,label = str(i))# 画线
            else:
                ax.plot(angles, data[i], linewidth=2,label=str(userId))
            ax.fill(angles, data[i], alpha=0.25)# 填充

        ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="SimHei")
        ax.set_title("学生"+str(userId)+"知识熟练度", va='bottom', fontproperties="SimHei")
        ax.set_rlim(0,10)
        ax.grid(True)
        plt.legend(loc="lower right")
        plt.savefig(saveFileName,format = 'png', dpi = 100)
        #plt.show()

    def plot_experimental(self, x, y, xlabel="time windows", ylabel=""):
        rate = 1.5
        xy_label_size = 20 * rate
        tar_size = 20 * rate
        tar_in_size = 10 * rate
        point_size = 12 * rate
        line_size = 2 * rate
        x_size = 5 * rate
        y_size = 4 * rate

        plt.figure(figsize=(x_size, y_size))
        plt.plot(x, y, label='TARE', marker='p', color='black', mfc='w', linewidth=line_size, markersize=point_size)
        plt.xlabel(xlabel, fontsize=tar_size)
        plt.ylabel(ylabel, fontsize=tar_size)
        plt.xticks(x, fontsize=xy_label_size)
        plt.yticks(fontsize=xy_label_size)
        plt.legend(loc="lower right", fontsize=tar_in_size)  # 图例
        # plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
        plt.tight_layout(pad=0);
        plt.show()

    def user_KT_plot(self, userId, timeList, gifFileName):
        png_fileList = []
        path_fig="./Figure"+'/'+self.DataName
        if not os.path.exists(path_fig):
            os.makedirs(path_fig)
        for i in timeList:
            png_fileList.append(path_fig +'/'+ str(userId) + str(i) + ".png")
            self.user_knowledge_plt([i], userId, png_fileList[-1])
        gifFileName='./Figure'+'/'+self.DataName+'/'+gifFileName
        create_gif(gifFileName, png_fileList)



kpt=KPT(DataNam,TmDir,timeLC,timdivid)
kpt.fit(5,0.01)
kpt.saveModel()
kpt.Experimental()
kpt.user_KT_plot(30,range(timdivid),"test.gif")

