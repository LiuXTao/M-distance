import numpy as np
import time
import pickle
import DataProcessToMatrix as DM
# 定义M-distance-based recommendation类
class MBR:
    def __init__(self,R=None,Test=None,testData=None,userNums=None,itemNums=None):
        self._P=R.copy()         # P为预测的评分矩阵
        self._R=R                # R为元用户评分矩阵
        self._Test=Test          # Test为测试评分矩阵
        self._testData=testData  # 测试数据数组
        self._Info=None           # 表示由三行向量，num,sum,r平均值组成的一个矩阵
        self._m=None              # 保存代表用户对电影是否有评分的0-1矩阵，0代表没有评分，1代表评过分
        self._userNum=userNums    # 用户数量
        self._itemNum=itemNums    # 项目数量（此处为电影数量)
        self._usingTime=0         # 表示预测耗费时间

    def mbr_predict(self,delta=0):
        # 使用self.m来保存代表用户对电影是否有评分的0-1矩阵，0代表没有评分，1代表评过分
        self._m=np.zeros((len(self._R),len(self._R[0])))
        self._m[self._R != 0] = 1

        # 开始训练时间
        startTime = time.time()

        sums = np.sum(self._R, axis=0)
        nums = np.sum(self._m, axis=0)
        nums[nums==0]=1e-15

        self._Info = np.array([nums, sums,sums/nums])
        for item in self._testData:
            i=item[0]-1            # 得到user index
            j=item[1]-1            # 得到item index

            if i>=self._userNum:
                continue

            # 步骤一：寻找临近点
            md=np.abs(self._Info[2]-self._Info[2][j])  # 得到平均距离差的绝对值
            a = np.zeros(len(md))
            a[md<=delta]=1
            a=np.multiply(a,self._m[i])    # 去除R[i][k]==0的情况
            nb=np.sum(a)                  # 得到邻居点数量
            nbsum=np.sum(a*self._R[i])     # 得到邻居点距离总合

            # 步骤二：更新信息
            # 计算预测值,并更新
            if nb >= 1:
                self._P[i][j] = int(nbsum/nb+0.5)
            else:
                self._P[i][j] = self._Info[2][j]
            # 更新num,sum,和r的平均值向量
            self._Info[0][j] +=1
            self._Info[1][j]+=self._P[i][j]
            self._Info[2][j]=self._Info[1][j]/self._Info[0][j]

        self._usingTime=time.time()-startTime
        print("MBR花费%f s\n" %(self._usingTime))
        return self._usingTime

    def get_MAE(self):
        n=np.zeros((len(self._Test),len(self._Test[0])))
        n[self._Test != 0] =1
        nonZero=np.sum(n.reshape(-1,1))
        mae=np.sum(np.abs(self._P*n-self._Test))/nonZero
        return mae

    def get_RMSE(self):
        n = np.zeros((len(self._Test), len(self._Test[0])))
        n[self._Test != 0] = 1
        nonZero = np.sum(n.reshape(-1, 1))
        rmse=np.sqrt(np.sum(np.square(self._P*n-self._Test))/nonZero)

        return rmse

#  MovieLen100k数据集测试程序入口
if __name__=="__main__":
    # 1.数据预处理
    # MovieLen100k数据集路径
    trainDataDir ="./data/u1.base"
    testDataDir="./data/u1.test"
    # 读取数据
    trainData=DM.readData(trainDataDir)
    testData=DM.readData(testDataDir)
    # 得到矩阵的user_nums和item_nums
    user_num=np.max(trainData,0)[0]
    item_num=np.max(trainData,0)[1]
    # 得到评分矩阵，此处train即是R矩阵，原始评分矩阵
    train=DM.toMatrix(trainData,user_num,item_num)
    test=DM.toMatrix(testData,user_num,item_num)

    # MovieLen100k用户数量
    user_list=[100,200,300,400,500,600,700,800,900]
    time_list=[]
    loss_list=[]
    # 2.定义训练器
    for i in user_list:
        mbr=MBR(R=train[0:i,:],Test=test[0:i,:],testData=testData,userNums=i,itemNums=item_num)
       # 开始预测
        usingTime=mbr.mbr_predict(delta=0.3)
        time_list.append(usingTime)
        mae=mbr.get_MAE()
        loss_list.append(mae)
        rmse=mbr.get_RMSE()
        print("mae loss: ",mae)
        print("rmse loss: ",rmse)
    #
    output=open("m_time.pkl",'wb')
    outputl=open("mloss.pkl",'wb')
    pickle.dump(time_list,output)
    pickle.dump(loss_list,outputl)

## EachMovie数据集测试程序入口
# if __name__ == "__main__":
#     # 1.数据预处理
#     # EachMovie数据集路径
#     trainDataDir = "./data2/u1.base"
#     testDataDir = "./data2/u1.test"
#     # 读取数据
#     trainData = DM.readData(trainDataDir)
#     testData = DM.readData(testDataDir)
#     # 得到矩阵的user_nums和item_nums
#     user_num = np.max(trainData, 0)[0]
#     item_num = np.max(trainData, 0)[1]
#     # 得到评分矩阵，此处train即是R矩阵，原始评分矩阵
#     train = DM.toMatrix(trainData, user_num, item_num)
#     test = DM.toMatrix(testData, user_num, item_num)
#
#     # EachMovie用户数量
#     user_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
#     time_list = []
#     loss_list = []
#     # 2.定义训练器
#     for i in user_list:
#         mbr = MBR(R=train[0:i, :], Test=test[0:i, :], testData=testData, userNums=i, itemNums=item_num)
#         # 开始预测
#         usingTime = mbr.mbr_predict(delta=0.3)
#         time_list.append(usingTime)
#         mae = mbr.get_MAE()
#         loss_list.append(mae)
#         rmse = mbr.get_RMSE()
#         print("mae loss: ", mae)
#         print("rmse loss: ", rmse)
#     #
#     output=open("m_timess.pkl",'wb')
#     outputl=open("mlossss.pkl",'wb')
#     pickle.dump(time_list, output)
#     pickle.dump(loss_list, outputl)












































































































