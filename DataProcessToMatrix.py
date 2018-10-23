import numpy as np
#定义读取数据的函数
def readData(dir):
    result=[]
    with open(dir,"r") as file:
        for line in file:          #从每一行读出数据，填入temp临时元组中
            temp=[int(line.split()[0]),int(line.split()[1]),int(line.split()[2])]
            result.append(temp)     #产生用于构造原始评分矩阵的数据组
    return result

#定义初始化矩阵函数，用于构造原型评分函数
def toMatrix(data,user_num,item_num):
    #定义矩阵维度
    matrixs=np.zeros((user_num,item_num))
    for i in data:            #i的形式为[i,j,value]
        matrixs[i[0]-1][i[1]-1]=i[2]
    return matrixs
