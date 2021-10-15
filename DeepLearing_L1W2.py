print("Hello world")

#!/usr/bin/env python
# coding: utf-8


test="Hello World"


print("test: "+test)

#numpy是Python中主要的科学计算包，对于math和numpy的对比。先用math
import math
def basic_sigmoid(x):
    s=1/(1+math.exp(-x))
    return s
basic_sigmoid(3)





#当输入的是矩阵或向量时，就需要遍历，用math调用不是很方便
x=[1,2,3]
for i in x:
    print(basic_sigmoid(i))





import numpy as np
x=np.array([1,2,3])    #x是行向量，调用numpy中的向量函数
print(np.exp(x))     #np.exp(x)会将指数函数应用于x的每个元素





#如果x是向量，则s=x+3或s=1/x之类的Python运算将输出与x维度大小相同的向量s
#x1=np.array(x)
print(x+4)





#numpy是Python中主要的科学计算包。用numpy
import numpy as np
def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s
sigmoid(x)





#编写一个梯度函数
def sigmoid_gradient(x):
    s=sigmoid(x)
    ds=s*(1-s)
    return ds
x=np.array([1,2,3])

print(sigmoid_gradient(x))
#当字符串和其他类型同时输出时，需要把sigmoid_gradient(x)转为str才能正确输出
print("sigmoid_gradient="+str(sigmoid_gradient(x)))





#重塑数组
#X.shape用于获取矩阵、向量的维度
#X.reshape()用于将X重塑为其他尺寸(第一维度image.shape[0],image.shape[1]...)





#例如当你读取由（length，height，depth=3）的3D数组表示的图像时，作为算法的输入时，会将3D重塑为1D向量，会将其转换为维度为（length*height*3,1）的向量
#将(a,b,c)的数组v重塑为维度(a*b,3)或(a*b*3,1)的向量
def image3D_1D(image):
    v=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    print(image.shape[0],image.shape[1],image.shape[2])
    return v





image=np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("image3D_1D(image)= "+str(image3D_1D(image)))




#行标准化 ，常用于对数据进行标准化。由于归一化后梯度下降的收敛速度更快，效果更好。将x更改为x/\\x\\，将每个x行向量除以其范数
def normalizeRows(x):
    x_norm=np.linalg.norm(x,axis=1,keepdims=True)
    x=x/x_norm
    return x
 





x=np.array([
    [0,3,4],
    [1,6,4]
    
])
print("normalizeRows(x)= "+str(normalizeRows(x)))





#广播和softmax函数.算法需要对两个或多个类进行分类时使用的标准化函数
def sofemax(x):
    x_exp=np.exp(x)
    x_sum=np.sum(x_exp,axis=1,keepdims=True)
    s=x_exp/x_sum
    return s
    





x=np.array([
    [9,2,5,0,0],
    [7,5,0,0,0]
])
print("sofemax(x)= "+str(sofemax(x)))

#-np.exp（x）适用于任何np.array x并将指数函数应用于每个坐标
#-sigmoid函数及其梯度
#-image2vector通常用于深度学习
#-np.reshape被广泛使用。 保持矩阵/向量尺寸不变有助于我们消除许多错误。
#-numpy具有高效的内置功能
#-broadcasting非常有用





#向量化
import time
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
tic=time.process_time()
dot=0
for i in range(len(x1)):
    dot+=x1[i]*x2[i]
toc=time.process_time()
print("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")





tic=time.process_time()
outer=np.zeros((len(x1),len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j]=x1[i]*x2[j]
toc=time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")




tic=time.process_time()
mul=np.zeros(len(x1))
for i in range(len(x1)):
    mul[i]=x1[i]*x2[i]
toc=time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")




W=np.random.rand(3,len(x1))
tic=time.process_time()
gdot=np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i]+=W[i,j]*x1[j]
toc=time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")





x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


#np.dot(x1,x2)   矩阵和矩阵的相乘
#np.outer(x1,x2)  第一个参数表示倍数，使得第二个向量每次变为几倍
#np.multiply(x1,x2)  矩阵点乘。如果shape不同的话，会将小规格的矩阵延展成与另一矩阵一样大小，再求两者内积。





#实现L1和L2损失函数，损失函数用于评估模型的性能。损失越大，预测与真实值的差异也就越大
#L1    采用方法——对\y^-y\求和
def L1(yhat,y):
    return np.sum(np.abs(y-yhat))
    #return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1= "+str(L1(yhat,y)))





#L2     采用方法——对\y^-y\的平方求和
def L2(yhat,y):
    #方法一
    #loss=np.dot((y-yhat),(y-yhat).T)   #行矩阵*列矩阵，为一个数。为该数的平方
    #return loss
    
    #方法二
    loss=(y-yhat)*(y-yhat)
    loss1=0
    for i in range(len(loss)):
        loss1+=loss[i]
    return loss1   

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2= "+str(L2(yhat,y)))


















