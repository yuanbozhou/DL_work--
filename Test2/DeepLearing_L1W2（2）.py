#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入数据
import h5py
#训练原始数据
train_data=h5py.File('C:/Users/fattysjm/Desktop/datasets/train_catvnoncat.h5','r')
test_data=h5py.File('C:/Users/fattysjm/Desktop/datasets/test_catvnoncat.h5','r')


# In[2]:


for key in train_data.keys():
    print(key)
    


# In[3]:


train_data['train_set_x'].shape
train_data['train_set_y'].shape


# In[4]:


for key in test_data.keys():
    print(key)


# In[5]:


test_data['test_set_x'].shape


# In[6]:


test_data['test_set_y'].shape


# In[7]:


#取出训练集 测试集
train_data_org=train_data['train_set_x'][:]
train_labels_org=train_data['train_set_y'][:]
test_data_org=test_data['test_set_x'][:]
test_labels_org=test_data['test_set_y'][:]


# In[8]:


#查看图片
import matplotlib.pyplot as plt
#在线显示图片
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(train_data_org[176])


# In[9]:


#数据维度的处理
m_train=train_data_org.shape[0]
m_test=test_data_org.shape[0]
#前面还为209，后面三列合为一列  ，再转置
train_data_tran=train_data_org.reshape(m_train,-1).T 
test_data_tran=test_data_org.reshape(m_test,-1).T 


# In[10]:


print(train_data_tran.shape,test_data_tran.shape)


# In[11]:


import numpy as np
train_labels_tran=train_labels_org[np.newaxis,:]
test_labels_tran=test_labels_org[np.newaxis,:]


# In[12]:


print(train_labels_tran.shape,test_labels_tran.shape)


# In[13]:


#标准化数据,像素是0—255，同除255,变为0-1，和0-1的标签值范围相同
train_data_sta=train_data_tran/255
test_data_sta=test_data_tran/255
#print(train_data_sta[9,9],train_test_sta)


# In[14]:


#定义sigmoid函数
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a


# In[15]:


#初始化参数
n_dim=train_data_sta.shape[0]
w=np.zeros((n_dim,1))  #零矩阵
b=0
#print(n_dim)


# In[16]:


# 定义前向传播函数，代价函数以及梯度下降
def propagate(w,b,X,y): 
    #1.前向传播函数
    z = np.dot(w.T,X)+b   #矩阵相乘用dot
    A = sigmoid(z)
    
    #2.代价函数
    m=X.shape[1]  #行是特征数，列是样本数
    J=-1/m*np.sum(y * np.log(A)+(1-y) * np.log(1-A))
    #axis(0/1)0为行，1为列\
    
    #3.梯度下降
    dw=1/m * np.dot(X,(A-y).T)
    db=1/m * np.sum(A-y )
    
    grands={'dw':dw,'db':db}   #对应key值存入grands字典中
    
    return grands,J


# In[17]:


#优化部分，迭代
def optimize(w,b,X,y,alpha,n_iters,print_cost):
    costs=[]  #空列表,为了记录值，后面画图需要值
    
    for i in range(n_iters):
        grands,J= propagate(w,b,X,y)
        #从字典中取出dw，db；
        dw=grands['dw']
        db=grands['db']
        
        w=w-alpha*dw
        b=b-alpha*db
        
        if i%100==0:
            costs.append(J)   #列表的追加
            if(print_cost):  #如果为true，打印出来；否则不打印
                print('n_iters is ',i,'cost is ',J)
     
    grands={'dw':dw,'db':db}   
    params={'w':w,'b':b}
          
    return grands,params,costs


# In[18]:


#预测部分
def predict(w,b,X_test):
    z = np.dot(w.T,X_test)+b
    A = sigmoid(z)
    
    m=X_test.shape[1]
    y_pred=np.zeros((1,m))
    
    for i in range(m):
        if A[:,i]>0.5:
            y_pred[:,i]=1
        else:
            y_pred[:,i]=0
    return y_pred
    


# In[19]:


#模型的整合（把优化和预测整合到一个部分里）
def model(w,b,X_train,y_train,X_test,y_test,alpha,n_iters,print_cost):
    grands,params,costs=optimize(w,b,X_train,y_train,alpha,n_iters,print_cost)
    w=params['w']   #从优化的结果中取出w，b
    b=params['b']
    
    y_pred_train=predict(w,b,X_train)  #在训练集上做预测
    y_pred_test=predict(w,b,X_test)  #在测试集上做预测
     
    #mean是准确率，*100是百分率
    print('the train acc is',np.mean(y_pred_train==y_train)*100,'%')
    print('the test acc is',np.mean( y_pred_test==y_test)*100,'%')
   
    d = {
        'w' : w,
        'b' : b,
        'costs':costs,
        'y_pred_train' : y_pred_train,
        'y_pred_test' : y_pred_test,
        'alpha' : alpha
    }
    
    return d
    


# In[20]:


#alpha为学习速率，n_iters为迭代次数
d =model(w,b,train_data_sta,train_labels_tran,test_data_sta,test_labels_tran,alpha=0.005,n_iters=2000,print_cost=True)


# In[21]:


plt.plot(d['costs'])
plt.xlabel('per hundred iters')
plt.ylabel('cost')




# In[22]:


index=45
print('y is ',test_labels_tran[0,index])
print('y prediction is ',int(d['y_pred_test'][0,index]))


# In[23]:


#print(test_data_tran[0,1])
plt.imshow(test_data_org[index])


# In[53]:


#比较alpha对于模型的影响
alpha=[0.01,0.001,0.0001]
for i in alpha:
    print('alpha= ',i)
    #d=model(w,b,train_data_sta,train_labels_tran,test_data_sta,test_labels_tran,i,n_iters=2000,print_cost=False)
    plt.plot(d['costs'],label=str(i))  #画出三条曲线
plt.xlabel('per hundred iters')
plt.ylabel('cost')
plt.legend()   #显示标签


# In[80]:


fname='C:\\Users\\fattysjm\\Desktop\\20211006212400.jpg'
image=plt.imread(fname)
plt.imshow(image)
#plt.waitforbuttonpress()


# In[81]:


#维度,尺寸>64，需要转换尺寸
image.shape


# In[82]:


#改变图片尺寸
from skimage import transform
image_tran=transform.resize(image,(64,64,3)).reshape(64*64*3,1)


# In[83]:


image_tran.shape


# In[84]:


y= predict(d['w'],d['b'],image_tran)  #返回最优参数里d里面的w和b
if int(y):
    print("暂时预测 是猫")
else:
    print("暂时预测 不是猫")


# In[ ]:





# In[ ]:




