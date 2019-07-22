#coding=utf-8
import numpy as np
import time
import tensorflow as tf
from OJDataProcessor import *
import pandas as pd
import matplotlib.pyplot as plt
from generateFigure import create_gif

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
    Limited=""

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
    # 读取R矩阵
    rmse=[]

    # 输入限制条件生成KPT_input.csv文件
    def ganeratedKPTData(self,Limited):
        tmp = OJDataProcessor(self.DataName, self.TmpDir)
        tmp.LoadSubmitRecordData(Limited[0], Limited[1],Limited[2], OnlyRight=True)
        tmp.loadSubmitRecordOfTimeWindows(Limited[2], Limited[3][0])
        tmp.loadQMatrix()
        print("Ganerated KPT_input.csv!")


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


	#读取R矩阵
    def readRmatrix(self):
        df = pd.read_csv(self.TmpDir + self.DataName +"_"+str(self.Limited[2])+"_"+str(self.Limited[3][0])+'_KPT_input.csv', index_col=None, header=None)
        R_matrix = np.zeros((df.loc[:, 0].max() + 1, df.loc[:, 1].max() + 1, self.num_item))
        df2 = df[df[3] == 1]
        for index, row in df2.iterrows():
            R_matrix[row[0]][row[1]][row[2]] = 1
        self.num_user = len(R_matrix[0])
        self.T = len(R_matrix)
        print('read Rmatrix completly!')
        return R_matrix

    # 读取Q矩阵
    def readQmatrix(self):
        Q_matrix = pd.read_csv(self.TmpDir + self.DataName + '_QMatrix.csv', index_col=0, header=0).as_matrix()
        Q_matrix = Q_matrix.T
        self.num_item = len(Q_matrix)
        self.K = len(Q_matrix[0])
        print('read Qmatrix completly!')
        return Q_matrix

    # 读取I矩阵
    def readImatrix(self):
        df = pd.read_csv(self.TmpDir + self.DataName + "_"+str(self.Limited[2])+"_"+str(self.Limited[3][0])+'_KPT_input.csv', index_col=None, header=None)
        I_matrix = np.zeros((df.loc[:, 0].max() + 1, df.loc[:, 1].max() + 1, self.num_item))
        for index, row in df.iterrows():
            I_matrix[row[0]][row[1]][row[2]] = 1
        print('read Imatrix completly!')
        return I_matrix


    #初始化
    def __init__(self,DataName= 'hdu',TmpDir='./data/',Limited=[[15,1000000,0.06,1],[10,1000000,0.02,1],['2018-11-22 20:45:48','2018-11-29 11:22:08'],[12]]):
        self.DataName = DataName
        self.TmpDir = TmpDir
        self.Limited=Limited
        self.ganeratedKPTData(Limited)
        self.R,self.I,self.Q_matrixs = self.loadData()

        #创建随机U矩阵 V矩阵
        self.U = tf.Variable(tf.random_normal([self.T,self.num_user,self.K], stddev=0.35, dtype=tf.float64))
        self.V = tf.Variable(tf.random_normal([self.num_item, self.K], stddev=0.35,dtype=tf.float64))
        self.alpha = tf.Variable(tf.random_normal([self.num_user,1], stddev = 0.35, dtype = tf.float64))
        print ("Ramdomly generate Q matrix!")
        K = self.K

        # 创建系数矩阵tmp矩阵
        # Q_matrixs [itemNum, K]
        # t1 [K, K*(K-1)] 
        t1 = np.zeros((K, K*(K-1)), dtype=float)
        for i in range(K):
            for j in range(K-1):
                t1[i][i*(K-1)+j] = 1;
                t1[j][i*(K-1)+j] = -1;
        # 创建偏序关系I矩阵
        self.tmp_all=tf.constant(t1)
        I_matrix = np.dot(self.Q_matrixs, t1)
        I_matrix = I_matrix > 0
        I_matrix = I_matrix.astype('float64')
        self.I_matrix = tf.constant(I_matrix)
        print ("Partial order relationship I_matrix generated!")
        print ("End of initialization!")


    def fit(self, epochs = 10, learnRate = 1e-3):
        print("start fitting")
        # 创建loss函数,这里创建了两个：1是涉及到时间因素的部分 2是涉及到Q先验的部分
        
        def loss_function1(Uba,t):
            if (Uba == 0):
                los = 1/2*tf.reduce_sum(self.I*(self.R[t]-tf.matmul(self.U[t],self.V,transpose_b=True))**2)
            else :
                los = 1/2*tf.reduce_sum(self.I*(self.R[t]-tf.matmul(self.U[t],self.V,transpose_b=True))**2+self.lambdau*tf.reduce_sum((Uba-self.U[t])**2))
            return los

        def loss_function2():
            los = -self.lambdap*tf.reduce_sum(self.I_matrix*tf.math.log(tf.sigmoid(tf.matmul(self.V,self.tmp_all))))+self.lambdav/2*tf.reduce_sum(self.V**2)
            return los

        def frequen_know():
            for l in open(self.TmpDir+self.DataName+'_'+str(self.Limited[2])+'_'+str(self.Limited[3][0])+"_KPT_input.csv", "r",encoding="UTF-8"):
                ls = l.strip().split(',')
                j = int(float(ls[0]))
                item_id = int(ls[2])
                for i in range(len(self.Q_matrixs[0])):
                    if self.Q_matrixs[item_id][i] == 1:
                        f_k[j][i] += 1
            return f_k

        def learn(Ut_1,t,f_k):
            l= Ut_1*(self.D*f_k[t])/(f_k[t]+self.r)
            return l

        def forgot(Ut_1):
            f = Ut_1*tf.to_double(tf.exp(-(self.delta_t)/self.S),name="ToDouble")
            return f

        # 优化器
        optimizer = tf.train.AdamOptimizer(learnRate)

        #梯度计算
        f_k = np.zeros((12,12))
        f_k = frequen_know()

        for epoch in range(epochs):
            loss=0
            start=time.time()
            # 12个时间点
            with tf.GradientTape() as tape:
                loss = loss_function2()
                #print (1, loss)
                for t in range(self.T):
                    if t==0:
                        Uba=0
                    else:
                        Uba=self.alpha*learn(self.U[t-1],t,f_k)+(1-self.alpha)*forgot(self.U[t-1])
                    #print (t, Uba)
                    loss += loss_function1(Uba, t) 
                    #print (t, loss_function1(Uba, t), loss)

            gradients = tape.gradient(loss, [self.U, self.V, self.alpha])
            optimizer.apply_gradients(zip(gradients, [self.U, self.V, self.alpha]))
            self.saveModel()
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,loss.numpy()/12))
            print('Time taken for one epoch {} sec\n'.format(time.time() - start))
        self.saveModel()
        print("fit completly")


    #self.TmpDir + self.DataName + "_"+str(self.Limited[2])+"_"+str(self.Limited[3][0])+'_KPT_Model'
    def saveModel(self):
        path = self.TmpDir + self.DataName + "_"+str(self.Limited[2])+"_"+str(self.Limited[3][0])+'_KPT_Model/KPT.npy' 
        U = self.U.numpy()
        V = self.V.numpy()
        alpha = self.alpha.numpy()
        np.save(path,np.array([U,V,alpha]))
        print ("KPT saved!")

    def loadModel(self):
        path = self.TmpDir + self.DataName + "_"+str(self.Limited[2])+"_"+str(self.Limited[3][0])+'_KPT_Model/KPT.npy' 
        [U, V, alpha] = np.load(path)
        self.U = tf.Variable(U)
        self.V = tf.Variable(V)
        self.alpha = tf.Variable(alpha)
        print ("load latest KPT Model")

    def plot_experimental(self,x,y,xlabel="time windows",ylabel=""):
        rate = 1.5
        xy_label_size = 20*rate
        tar_size = 20*rate
        tar_in_size = 10*rate
        point_size = 12*rate
        line_size = 2*rate
        x_size = 5*rate
        y_size = 4*rate

        plt.figure(figsize=(x_size,y_size))
        plt.plot(x,y,label='TARE',marker = 'p', color='black',mfc='w',linewidth=line_size,markersize = point_size) 
        plt.xlabel(xlabel, fontsize=tar_size) 
        plt.ylabel(ylabel, fontsize=tar_size) 
        plt.xticks(x, fontsize=xy_label_size)
        plt.yticks(fontsize=xy_label_size)
        plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
        #plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
        plt.tight_layout(pad=0);
        plt.show() 


    def Experimental(self):
        def RMSE(R, _R):
            ans = tf.sqrt(tf.reduce_sum((_R - R)**2)/(self.num_user * self.num_item))
            return ans

        def MAE(R, _R):
            ans = tf.reduce_sum(tf.abs(_R-R))/(self.num_user * self.num_item)
            return ans

        RMSE_list = []
        for i in range(self.T-1):
            RMSE_val = RMSE(tf.matmul(self.U[i], self.V,transpose_b = True), self.R[i+1])
            RMSE_list.append(RMSE_val.numpy())
            print ("time windows Id:", i+2, "  RMSE:",RMSE_list[-1])

        MAE_list = []
        for i in range(self.T-1):
            MAE_val = MAE(tf.matmul(self.U[i], self.V,transpose_b = True), self.R[i+1])
            MAE_list.append(MAE_val.numpy())
            print ("time windows Id:", i+2, "  MAE:",MAE_list[-1])
        x = range(2,self.T+1)
        self.plot_experimental(x,RMSE_list,"time windows","RMSE")
        self.plot_experimental(x,MAE_list,"time windows","MAE")
        return RMSE_list, MAE_list

    def user_knowledge_plt(self, timeList, userId, saveFileName = "./test.png", knowledgeLabel = ['基础题','动态规划','搜索','贪心','递推','数学题','字符串','大数','博弈','母函数','哈希','无章法']):
        #数据
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
                ax.plot(angles, data[i], linewidth=2)
            ax.fill(angles, data[i], alpha=0.25)# 填充

        ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="SimHei")
        ax.set_title("学生"+str(userId)+"知识熟练度", va='bottom', fontproperties="SimHei")
        ax.set_rlim(0,10)
        ax.grid(True)
        plt.legend(loc="lower right")
        plt.savefig(saveFileName,format = 'png', dpi = 100)
        #plt.show()

    def user_KT_plot(self, userId, timeList, gifFileName):
        png_fileList = []
        for i in timeList:
            png_fileList.append("./Figure/" + str(userId) + str(i) + ".png") 
            self.user_knowledge_plt([i], userId, png_fileList[-1])
        create_gif(gifFileName, png_fileList)

#kpt = KPT()
#kpt.fit(100,0.01) 
#kpt.saveModel()
#kpt.loadModel()
#kpt.fit(200,0.01)
#kpt.saveModel()
#kpt.Experimental()

#kpt.user_KT_plot(30,range(12),"./test.gif")

