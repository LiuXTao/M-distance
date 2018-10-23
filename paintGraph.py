import pickle
import matplotlib.pyplot as plt

#time
file1=open('m_time.pkl', 'rb')
m_time=pickle.load(file1)
file2=open('pnn_time.pkl','rb')
p_time=pickle.load(file2)
file3=open('cnn_time.pkl','rb')
c_time=pickle.load(file3)

#loss
lfile1=open("mloss.pkl",'rb')
m_loss=pickle.load(lfile1)
lfile2=open("pnn_loss.pkl",'rb')
p_loss=pickle.load(lfile2)
lfile3=open("cnn_loss.pkl",'rb')
c_loss=pickle.load(lfile3)


user_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]
# runtime比较
plt.figure(1)
plt.title('The time comparison')
plt.xlabel('Users')
plt.ylabel('Times(s)')
plt.plot(user_list,m_time,label="MBR")
plt.plot(user_list,p_time,label="PNN")
plt.plot(user_list,c_time,label="CNN")
plt.legend()

# MAE 误差比较
plt.figure(2)
plt.title('The MAE comparison')
plt.xlabel('Users')
plt.ylabel('Mae')
plt.plot(user_list,m_loss,label="MBR")
plt.plot(user_list,p_loss,label="PNN")
plt.plot(user_list,c_loss,label="CNN")
plt.legend()


user_lists=[10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
#显示EachMovie数据集运行时间结果图
filess=open("m_timess.pkl","rb")
m_times=pickle.load(filess)
plt.figure(3)
plt.title('The runtime of ')
plt.xlabel('Users')
plt.ylabel('Times(s)')
plt.plot(user_lists,m_times,label="MBR")
plt.legend()

plt.show()