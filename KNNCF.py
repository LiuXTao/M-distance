from math import sqrt
import time
import pickle
# 读取数据函数
def loadData(TrainFile,TestFile):
    trainSet = {}
    testSet = {}
    movieUser = {}
    u2u = {}
    # 加载训练集
    for line in open(TrainFile):
        (userId, itemId, rating, timestamp) = line.strip().split('\t')
        trainSet.setdefault(userId,{})
        trainSet[userId].setdefault(itemId,float(rating))
        movieUser.setdefault(itemId,[])
        movieUser[itemId].append(userId.strip())
    # 加载测试集
    for line in open(TestFile):
        (userId, itemId, rating, timestamp) = line.strip().split('\t')
        testSet.setdefault(userId,{})
        testSet[userId].setdefault(itemId,float(rating))
    # 生成用户共有电影矩阵
    for m in movieUser.keys():
        for u in movieUser[m]:
            u2u.setdefault(u,{})
            for n in movieUser[m]:
                if u!=n:
                    u2u[u].setdefault(n,[])
                    u2u[u][n].append(m)
    return trainSet,testSet,u2u

class KNN:
    def __init__(self,trainSet=None,testSet=None,u2u=None,K=1,cla="P",userNums=0):
        self.trainSet=trainSet   # 训练数据
        self.testSet=testSet     # 测试数据
        self.u2u=u2u             # 用户共有电影矩阵
        self.K=K                # 表示K个临近点为一簇
        self.usingTime=0
        self.userSim=None
        self.cla=cla
        self.userNums=userNums
        if cla=="P":
            self.similarity=self.peasronSimilarity
        if cla=="C":
            self.similarity=self.cosineSimilarity


    # 计算一个用户的平均评分
    def getAverageRating(self,user):

        average = (sum(self.trainSet[user].values())*1.0) / len(self.trainSet[user].keys())
        return average

    # 计算用户相似度
    def getUserSim(self):
        userSim = {}
        # 计算用户的用户相似度
        for u in self.u2u.keys():             # 对每个用户u
            userSim.setdefault(u,{})  # 将用户u加入userSim中设为key，该用户对应一个字典
            average_u_rate = self.getAverageRating(u)  # 获取用户u对电影的平均评分
            for n in self.u2u[u].keys():  # 对与用户u相关的每个用户n
                if int(n)>self.userNums:
                    continue

                userSim[u].setdefault(n,0)  # 将用户n加入用户u的字典中
                average_n_rate = self.getAverageRating(n)  # 获取用户n对电影的平均评分

                userSim=self.similarity(userSim,u,n,average_u_rate,average_n_rate)

        return userSim
    # 皮尔森距离
    def peasronSimilarity(self,userSim,u,n,average_u_rate,average_n_rate):
        subsquare = 0   # 皮尔逊相关系数的分子部分
        fenmu1 = 0      # 皮尔逊相关系数的分母的一部分
        fenmu2 = 0      # 皮尔逊相关系数的分母的一部分
        for m in self.u2u[u][n]:  # 对用户u和用户n的共有的每个电影
            subsquare += (self.trainSet[u][m] - average_u_rate) * (self.trainSet[n][m] - average_n_rate) * 1.0
        fenmu1 += pow(self.trainSet[u][m] - average_u_rate, 2) * 1.0
        fenmu2 += pow(self.trainSet[n][m] - average_n_rate, 2) * 1.0

        fenmu1 = sqrt(fenmu1)
        fenmu2 = sqrt(fenmu2)

        if fenmu1 == 0 or fenmu2 == 0:  # 若分母为0，相似度为0
            userSim[u][n] = 0
        else:
            userSim[u][n] = subsquare / (fenmu1 * fenmu2)
        return userSim
    # 余弦距离
    def cosineSimilarity(self,userSim,u,n,average_u_rate,average_n_rate):
        product = 0  # 余弦相关系数的分子部分
        fenmu1 = 0  # 余弦相关系数的分母的一部分
        fenmu2 = 0  # 原先相关系数的分母的一部分

        for m in self.u2u[u][n]:
            product+=self.trainSet[u][m]*self.trainSet[n][m] * 1.0
        fenmu1+=pow(self.trainSet[u][m],2)*1.0
        fenmu2+=pow(self.trainSet[n][m],2)*1.0
        fenmu1=sqrt(fenmu1)
        fenmu2=sqrt(fenmu2)
        if fenmu1==0 or fenmu2==0:
            userSim[u][n]=0
        else:
            userSim[u][n]=product/(fenmu1*fenmu2)
        return userSim


    # 寻找用户最近邻并生成推荐结果
    def getRecommendations(self):
        startTime = time.time()
        self.userSim=self.getUserSim()
        pred = {}
        for user in self.trainSet.keys():    # 对每个用户


            pred.setdefault(user,{})    # 生成预测空列表
            interacted_items = self.trainSet[user].keys() # 获取该用户评过分的电影
            average_u_rate = self.getAverageRating(user)  # 获取该用户的评分平均分
            userSimSum = 0
            simUser = sorted(self.userSim[user].items(),key = lambda x : x[1],reverse = True)[0:self.K]
            for n, sim in simUser:
                average_n_rate = self.getAverageRating(n)
                userSimSum += sim   # 对该用户近邻用户相似度求和
                for m, nrating in self.trainSet[n].items():
                    if m in interacted_items:
                        continue
                    else:
                        pred[user].setdefault(m,0)
                        pred[user][m] += (sim * (nrating - average_n_rate))
            for m in pred[user].keys():
                    pred[user][m] = average_u_rate + (pred[user][m]*1.0) / userSimSum
        self.usingTime = time.time() - startTime
        print("用时 ", self.usingTime)
        return pred,self.usingTime

    #计算预测分析准确度
    def get_MAE(self,pred):
        MAE = 0
        rSum = 0
        setSum = 0
        for user in pred.keys():                        #对每一个用户
            for movie, rating in pred[user].items():    #对该用户预测的每一个电影
                if user in self.testSet.keys() and movie in self.testSet[user].keys() :   #如果用户为该电影评过分
                    setSum = setSum + 1                 #预测准确数量+1
                    rSum = rSum + abs(self.testSet[user][movie]-rating)      #累计预测评分误差
        MAE = rSum / setSum
        return MAE

# 主程序入口
if __name__ == '__main__':
    # 1.读取数据
    trainSet, testSet, u2u = loadData(TrainFile="./data/u1.base",TestFile="./data/u2.base")
    print(type(u2u))

    user_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    # 2.PNN测试
    print("PNN测试----")
    output1=open("pnn_time.pkl",'wb')
    output11 = open("pnn_loss.pkl", 'wb')
    K = 30
    ptime_list = []
    ploss_list=[]
    for i in user_list:
        pNN=KNN(trainSet={key: value for key, value in trainSet.items() if int(key) <=i},
                testSet={key: value for key, value in testSet.items() if int(key) <=i},
                u2u={key: value for key, value in u2u.items() if int(key) <=i},K=K,cla="P",userNums=i)
        pred,usingTime = pNN.getRecommendations()    # 获得推荐
        ptime_list.append(usingTime)
        mae = pNN.get_MAE(pred)                      # 计算MAE
        ploss_list.append(mae)
        print (u'邻居数为：N= %d 时 预测评分准确度为：MAE=%f'%(K,mae))
    pickle.dump(ptime_list,output1)
    pickle.dump(ploss_list,output11)
    #CNN测试
    print("CNN测试----")
    output2 = open("cnn_time.pkl", 'wb')
    output22=open("cnn_loss.pkl",'wb')
    ctime_list=[]
    closs_list=[]
    for i in user_list:
        cNN=KNN(trainSet={key: value for key, value in trainSet.items() if int(key) <=i},
                testSet={key: value for key, value in testSet.items() if int(key) <=i},
                u2u={key: value for key, value in u2u.items() if int(key) <=i},K=K,cla="C",userNums=i)
        pred,usingTime = cNN.getRecommendations()    # 获得推荐
        ctime_list.append(usingTime)
        mae = cNN.get_MAE(pred)                      # 计算MAE
        closs_list.append(mae)
        print (u'邻居数为：N= %d 时 预测评分准确度为：MAE=%f'%(K,mae))
    pickle.dump(ctime_list,output2)
    pickle.dump(closs_list,output22)