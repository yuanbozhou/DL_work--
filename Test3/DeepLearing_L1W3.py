#!/usr/bin/env python
# coding: utf-8

# In[1]:


#错误笔记：
#1.最终运行时应把前面调用函数的测试部分全部注释掉，避免测试部分使用的参数和最终运行的
# 传入的参数重复，造成错误。


# In[2]:


get_ipython().run_line_magic('pwd', '')


# In[3]:


# cd E:\\jupyterCode


# In[4]:


# cd .ipynb_checkpoints


# In[5]:


#sklearn提供了用于数据挖掘和分析的简单有效的工具
#matplotlib是Python中常用的绘制图形的库
#testCases提供了一些测试示例用以评估函数的正确性
#planar_utils提供了此作业中使用的各种函数


# In[6]:


#1---------安装包
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# In[31]:


#2-------------数据集①
#以下代码会将“flower” 2分类数据集加载到变量 X 和 Y中。
X,Y=load_planar_dataset()
#红色（y=0）和蓝色（y=1），使用matplotlib可视化数据集
plt.scatter(X[0,:],X[1,:],c=Y.reshape(X[0,:].shape),s=40,cmap=plt.cm.Spectral)


# In[8]:


#2-------------数据集②
"""
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
dataset = "noisy_moons"
X,Y=datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
if dataset == "blobs":
    Y = Y % 2

#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
#上一语句如出现问题请使用下面的语句：
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
"""


# In[9]:


#现在有的：
#包含特征（x1，x2）的numpy数组（矩阵）X
#包含标签（红色：0，蓝色：1）的numpy数组（向量）Y。

#########################
#了解数据集：
#1.数据集中有多少个训练示例？
#2.变量“X”和“Y”的shape是什么？


# In[10]:


shape_X=X.shape
shape_Y=Y.shape
m=shape_X[1]
print('The shape of X is: '+str(shape_X))
print('The shape of Y is: '+str(shape_Y))
print('I have m = %d training examples!' % (m))


# In[11]:


#3------------简单的Logistic回归
#可以使用sklearn的内置函数来执行此操作，运行以下代码在数据集上训练逻辑回归分类器
clf=sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T,Y.T)


# In[32]:


#运行下面的代码绘制此模型的决策边界：
# 错误 plot_decision_boundary(lambda x:clf.predict(x),X,Y)   
#squeeze()函数的功能是：从矩阵shape中，去掉维度为1的。
plot_decision_boundary(lambda x:clf.predict(x), X, np.squeeze(Y)) #绘制决策边界
plt.title("Logistic Regression")    #图标题
LR_predictions=clf.predict(X.T)     #预测结果
print("逻辑回归的准确性： %d " % float((np.dot(Y,LR_predictions)+np.dot(1 - Y,1 - LR_predictions))/float(Y.size)*100)+"% " + "(正确标记的数据点所占的百分比)")
#print(np.dot(Y,LR_predictions))
#print(np.dot(1 - Y,1 - LR_predictions))
#print(float(Y.size))
#print(Y)
#print(LR_predictions)
#print(Y.shape)
#print(LR_predictions.shape)


# In[13]:


#准确性只有47%的原因是数据集不是线性可分的，所以逻辑回归表现不佳，现在我们正式开始构建神经网络.
#采用激活函数a=tan（z）
#建立神经网络的一般方法是：
# 1.定义神经网络结构（输入单元的数量，隐藏单元的数量）
# 2.初始化模型的参数
# 3.循环：
#     实施前向传播
#     计算损失
#     实现向后传播
#     更新参数（梯度下降）
# 我们要它们合并到一个nn_model()函数中，
#当我们构建好了nn_model()并学习了正确的参数，我们就可以预测新的数据


# In[14]:


# 1.定义神经网络结构（输入单元的数量，隐藏单元的数量）
"""
 n_x:输入层的数量
 n_h：隐藏层的数量（本实验设置为4）
 n_y：输出层的数量
"""
def layer_sizes(X,Y):
    """
    参数：
     X - 输入数据集,维度为（输入的数量，训练/测试的数量）
     Y - 标签，维度为（输出的数量，训练/测试数量）
    
    返回：
     n_x - 输入层的数量
     n_h - 隐藏层的数量
     n_y - 输出层的数量
    """
    n_x=X.shape[0]  #输入层
    n_h=4 #隐藏层，硬编码为4
    n_y=Y.shape[0] #输出层
    
    return (n_x,n_h,n_y)
"""
#测试layer_sizes
print("==========layer_sizes==========")
X_asses,Y_asses=layer_sizes_test_case()
(n_x,n_h,n_y)=layer_sizes(X_asses,Y_asses)
print("输入层的节点数量为：n_x="+str(n_x))
print("隐藏层的节点数量为：n_h="+str(n_h))
print("输出层的节点数量为：n_y="+str(n_y))
"""


# In[15]:


# 2.初始化模型的参数
"""
np.random.randn(a,b)*0.01来随机初始化一个维度为（a，b）的矩阵
理解：
W是随机生成了符合高斯分布的数，再*0.01，使得W很小，
为何要随机生成？ 因为如果像b一样全赋0值，就会产生对称性，假如hidden layer有两个由输入值产生的神经元，则由于对称性，无论是正向传播还是反向传播，这两个本应表示不同特征的神经元所做的运算都是一致的，即对称的，not well，所以随机生成了一些数赋值给矩阵W，计算b就不用担心对称性的问题，故直接赋0值
而*0.01使得W很小是因为，可以参照激活函数sigmoid和tanh，当W很大，用W * X+b=a得到的a很大，再用对a用激活函数如sigmoid(a)，由于a很大了，sigmoid(a)中的a会趋向正无穷或负无穷，则函数值sigmoid(a)趋向于一个平缓的趋势，在梯度下降的时候计算的梯度很小，会导致学习的很慢，故使得W取一个很小的值(激活函数图sigmoid,tanh在网上可以很容易找到)。不过在某些情况下不取0.01.会取其他的比较小的值

np.zeros((a,b)) 用零初始化矩阵（a，b）
"""

def initialize_parameters(n_x,n_h,n_y):
    """
    参数：
     n_x - 输入层节点的数量
     n_h - 隐藏层节点的数量
     n_y - 输出层节点的数量
    返回：
    parameters - 包含参数的字典：
    W1-权重矩阵，维度为(n_h,n_x)
    b1-偏向量，维度为(n_h,1)
    w2-权重矩阵，维度为(n_y,n_h)
    b2-偏向量，维度为(n_y,1)
    
    """
    np.random.seed(2)#指定一个随机种子，以便于你的输出与我们的一样。
    """
    我们在调用random.rand()时，每次产生的数都是随机的。但是，当我们预先
    使用 random.seed(x) 设定好种子之后，其中的 x 可以是任意数字，
    如10，这个时候，先调用它的情况下，使用 random() 生成的随机数将会是同一个。
    """
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros(shape=(n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros(shape=(n_y,1))
    
    #使用断言确保我的数据格式是正确的
    assert(W1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))
    
    #用字典储存，这样返回字典，从字典中参数值获取value。就不用返回一大堆了。
    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2        
    }
    
    return parameters
"""    
#测试initialize_parameters
print("==========initialize_parameters==========")
#X_asses,Y_asses=layer_sizes_test_case()
#(n_x,n_h,n_y)=layer_sizes(X_asses,Y_asses)

n_x , n_h , n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x , n_h , n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))  
"""


# In[16]:


# 3.循环：
    # --实施前向传播。
    # 我们可以使用sigmoid()函数，也可以使用np.tanh()函数。
    # 这里我们用np.tanh()函数

def forward_propagation(X,parameters):
    """
    参数：
        X- 维度为（n_x,m）的输入数据 ,因为有m个训练集
        parameters -初始化函数(initialize_parameters)的输出
    返回：
        A2-使用sigmoid（）函数计算的第二次激活后的数值
        cache-包含"Z1","A1","Z2","A2"的字典类型数量
    
    """
    W1= parameters["W1"]             #注意是[],不是()
    b1= parameters["b1"]
    W2= parameters["W2"]
    b2= parameters["b2"]
    
    #前向传播计算A2
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    #A2=np.tanh(Z2)
    #使用断言确保我的数据格式是正确的
    assert(A2.shape==(1,X.shape[1])) #?????????为什么是X.shape[1]
    #assert(A2.shape==(n_y,1)) #不能用这个，因为n_y不是传入的参数了
    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }
    
    return (A2,cache)
"""    
#测试forward_propagation
print("==========forward_propagation==========")
X,parameters = forward_propagation_test_case()
A2,cache = forward_propagation(X,parameters)
#求其中一个用A2 =forward_propagation(X,parameters)[0]

print("Z1 = " + str(cache["Z1"]))
print("A1 = " + str(cache["A1"]))
print("Z2 = " + str(cache["Z2"]))
print("A2 = " + str(cache["A2"]))  

#numpy.mean(a, axis, dtype, out，keepdims )函数。本次实验选择返回实数的。

axis 不设置值，对 m*n 个数求均值，返回一个实数
axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
"""
"""
print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))    
"""    
    
    
    
    


# In[17]:


# 3.循环：
    # --计算损失（损失函数/代价函数）
def compute_cost(A2,Y,parameters):
        """
        参数：
        A2-使用sigmoid（）函数计算的第二次激活后的数值
        Y-“True”标签向量，维度为（1，数量）
        parameters-一个包含W1，b1，W2和b2的字典类型的变量
        返回：
        成本-交叉熵成本给定的方程
    
        """
        m=Y.shape[1]
        W1=parameters["W1"]
        W2=parameters["W2"]
        
        #计算成本
        logprobs=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)
        cost=-(np.sum(logprobs))/m
        cost=float(np.squeeze(cost))
        #使用断言确保我的数据格式是正确的 
        """
        isinstance(object,classinfo)
        Python中的 isinstance() 函数，是Python中的一个内置函数，用来判断一个函数是否是一个已知的类型
        
        """
        assert(isinstance(cost,float)) 
        
        return cost
"""
#测试compute_cost
print("==========compute_cost==========")
A2 , Y_assess , parameters = compute_cost_test_case()
cost = compute_cost(A2 , Y_assess , parameters)
print(str(cost))       
"""        


# In[18]:


0.6929198937761266


# In[19]:


# 3.循环：
    # --实现向后传播 
    #tanh（a）是激活函数，求导后为1-a^2
    #
def backward_propagation(parameters,cache,X,Y):
        """
        使用六个方程来搭建反向传播函数。
        参数：
        parameters-包含W1，W2的字典
        cache-包含Z1，A1，Z2，A2的字典
        X-输入数据，维度为（2，数量）
        Y-“True”标签，维度为（1，数量）
        
        返回：
        grads-包含W和b的字典
        """
        m=X.shape[1]
        
        W1=parameters["W1"]
        W2=parameters["W2"]
        A1=cache["A1"]
        A2=cache["A2"]
        
#np.sum(A, axis = 1, keepdims = True)axis=1 以竖轴为基准 ，同行相加keepdims主要用于保持矩阵的二维特性
        dZ2=A2-Y
        dW2=(1/m)*np.dot(dZ2,A1.T)
        db2=(1/m)*(np.sum(dZ2,axis=1,keepdims=True))
        dZ1=np.dot(W2.T,dZ2)*(1-np.power(A1,2))
        dW1=(1/m)*np.dot(dZ1,X.T)
        db1=(1/m)*(np.sum(dZ1,axis=1,keepdims=True))
        grads={
            "dW2":dW2,
            "db2":db2,
            "dW1":dW1,
            "db1":db1
        }
        
        return grads
"""        
#测试backward_propagation
print("==========backward_propagation==========")
parameters,cache,X,Y= backward_propagation_test_case()
grads = backward_propagation(parameters,cache,X,Y)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))  
"""        


# In[20]:


# 3.循环：
    # --更新参数（梯度下降）。
def update_parameters(parameters,grads,learning_rate=5.1):
    """
    使用给出的梯度下降更新规则，更新参数
        参数：
        parameters-包含W1，W2的字典
        grads-包含dW1,db1,dW2,db2的字典
        learning_rate-学习速率
        返回：
        parameters-包含更新参数的字典类型的变量。
        
    """
    W1,W2= parameters["W1"], parameters["W2"]
    b1,b2= parameters["b1"], parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    #使用更新规则方程
    
    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2
    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    
    return parameters
"""    
#测试update_parameters
print("==========backward_propagation==========")
parameters,grads= update_parameters_test_case()
parameters = update_parameters(parameters,grads)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))      
"""    


# In[21]:


# 我们要它们合并到一个nn_model()函数中，当我们构建好了nn_model()并学习了正确的参数，
#我们就可以预测新的数据。
#神经网络模型必须以正确的顺序使用先前的功能
"""
定义的函数
layer_sizes (X,Y)
initialize_parameters (n_x,n_h,n_y)
forward_propagation (X,parameters)
compute_cost(A2,Y,parameters)
backward_propagation(parameters,cache,X,Y) 
update_parameters(parameters,grads,learning_rate)
""" 
def nn_model(X,Y,n_h,num_iterations = 10000,print_cost=False):
    """
    参数：
    X-数据集，维度为（2，示例数）
    Y-标签，维度为（1，示例数）
    n_h-隐藏层的数量
    num_iterations-梯度下降循环中的迭代次数
    print_cost-如果为True，则每1000次迭代打印一次成本数值(代价函数)
    返回：
    parameters-模型学习的参数，它们可以用来进行预测。    
    """
    
    np.random.seed(3) 
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]
    
    parameters=initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0,num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backward_propagation(parameters,cache,X,Y) 
        parameters=update_parameters(parameters,grads)
        
        if print_cost and i%1000==0:
                print("第",i,"次循环，成本为: "+str(cost))
                
    return parameters
"""
#测试nn_model
print("==========nn_model==========")
X,Y= nn_model_test_case()
parameters = nn_model(X,Y,4,num_iterations=10000,print_cost=False)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))    
"""


# In[33]:


#预测predict()来使用模型进行预测，使用前向传播来预测结果
def predict(parameters,X):
    """
    使用学习的参数，为X中的每个示例预测一个类
    
    参数：
    parameters-包含参数的字典
    X-输入数据(n_x,m) n_x个特征，m个样本即示例
    
    返回：
    predictions-我们模型预测的向量(红色：0/蓝色：1)
    
    """
    A2,cache=forward_propagation(X,parameters)
    #np.round()函数的做用：对给定的数组进行四舍五入。
    #但是需要特别注意的是,当整数部分以0结束时,round函数一律是向下取整。
    predictions=np.round(A2)
    
    return predictions
"""
#测试perdict
print("==========perdict==========")
parameters,X= predict_test_case()
predictions = predict(parameters,X)
# X以m * n矩阵举例
# np.mean(X) 矩阵中所有元素求均值
# np.mean(X,0) 压缩行，对各列求均值,返回 1* n 矩阵
# np.mean(X,1) 压缩列，对各行求均值,返回 m *1 矩阵
print("预测的平均值= "+str(np.mean(predictions)))
"""


# In[34]:


parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
plt.title("Decision Boundary for hidden layer size " + str(4))    #图标题
plt.show()
predictions = predict(parameters, X)     #预测结果
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')


# In[35]:


#更改隐藏层节点数量
#我们上面的实验把隐藏层定为4个节点，现在我们更改隐藏层里面的节点数量，看一看节点数量是否会对结果造成影响
plt.figure(figsize=(16,32))
hidden_layer_sizes=[1,2,3,4,5,20,50] #隐藏层的数量
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
for i,n_h in enumerate(hidden_layer_sizes):
    #位置是由三个整型数值构成,第一个代表行数,第二个代表列数,第三个代表索引位置
    #这个函数用来表示把figure分成nrows*ncols的子图表示， 索引值，表示把图画在第plot_number个位置（从左下角到右上角）
    plt.subplot(5,2,i+1)
    plt.title("Hidden Layer of size %d" % n_h)
    parameters=nn_model(X,Y,n_h,num_iterations=5000)
    plot_decision_boundary(lambda x:predict(parameters,x.T),X,np.squeeze(Y))
    predictions=predict(parameters,X)
    accuracy=float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print("隐藏层的节点数量：{} ，准确率{} %".format(n_h,accuracy))


# In[36]:


"""
结论：
较大的模型（具有更多隐藏单元）能够更好地适应训练集，直到最终的最大模型过度拟合数据。
最好的隐藏层大小似乎在n_h = 5附近。实际上，这里的值似乎很适合数据，而且不会引起过度拟合。
我们还将在后面学习有关正则化的知识，它允许我们使用非常大的模型（如n_h = 50），而不会出现太多过度拟合。

"""


# In[26]:


##【可选】探索
"""
1. 当改变sigmoid激活或ReLU激活的tanh激活时会发生什么？
2. 改变learning_rate的数值会发生什么
3. 如果我们改变数据集呢？
"""


# In[27]:


#1. 当改变sigmoid激活或ReLU激活的tanh激活时会发生什么？
#1.1 当改变sigmoid激活为tanh激活时：准确率降低为89%
#1.2 当改变sigmoid激活为tanh激活时：


# In[28]:


#2. 改变learning_rate的数值会发生什么（learning_rate为1.2， 90%）
#2.1 learning_rate为0.1  89%
#2.2 learning_rate增大为5.1 也为90%


# In[29]:


#3. 如果我们改变数据集呢？
#逻辑回归正确性为87%
#------------>>>>单隐藏层，隐藏节点数量为4 的决策边界提高为96%
#------------>>>>单隐藏层，隐藏节点数量为20 的决策边界提高为97%


# In[30]:


#知识点
"""
1.tanh激活函数通常比隐藏层单元的sigmoid激活函数效果更好，因为其输出的平均值更接近于零，因此它将数据集中在下一层是更好的选择
2.您正在构建一个识别黄瓜（y = 1）与西瓜（y = 0）的二元分类器。 你会推荐哪一种激活函数用于输出层？
sigmoid函数,因为sigmoid函数的输出值可以很容易地理解为概率。
3.你已经为所有隐藏的单位建立了一个使用tanh激活的网络。使用np.random.randn(…, …)*1000将权重初始化为相对较大的值。会发生什么？
tanh对于较大的值变得平坦，这导致其梯度接近于零。 这减慢了优化算法。这将导致tanh的输入也非常大，从而导致梯度接近于零。因此，优化算法将变得缓慢。

"""


# In[ ]:





# In[ ]:




