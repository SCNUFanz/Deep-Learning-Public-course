import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from sklearn import datasets
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import operator

x_data = datasets.load_iris().data     #获取iris数据集数据
y_data = datasets.load_iris().target   #获取iris数据标签

#随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
np.random.seed(120)
np.random.shuffle(x_data)
np.random.seed(120)
np.random.shuffle(y_data)
tf.random.set_seed(120)

#将打乱后的数据集分割为训练集和测试集
#训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

#转化x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)

#把数据集分批次，每个批次batch组数据
#from_tensor_sices函数使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

def poj(xt):
    #生成神经网络的参数，4个输入特征，故输入层为4个输入节点；因为分3类，故输出层为3个神经元
    #用tf.Vairable()标记参数可训练
    #使用seed使每次生成的随机数相同
    w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
    b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

    lr = 0.1  #学习率
    epoch = 250               #循环250次
    test_acc = []             #将每次的精确率记录在此列表中，为后续画acc曲线提供依据
    train_loss_results = []   #将每次的loss记录在此列表中，为后续画loss曲线提供依据
    loss_all = 0              #每轮分为4个step，loss_all记录4个step生成的4个loss的和

    #训练
    for epoch in range(1, epoch + 1):       #大循环，循环一次数据集
        for step,(x_train,y_train) in enumerate(train_db):      #小循环，循环取一个batch（32个数据集）
            with tf.GradientTape() as tape:          #with结构记录梯度信息
                y = tf.matmul(x_train,w1) + b1         #神经网络记录乘加运算
                y = tf.nn.softmax(y)                 #使y满足概率分布
                y_ = tf.one_hot(y_train,depth=3)     #3分类
                loss = tf.reduce_mean(tf.square(y_-y))   #预测值-真实值
                loss_all += loss.numpy()
            #求导
            grads = tape.gradient(loss,[w1,b1])
            #梯度自更新  w1=w1-lr*w1_grad  b1=b1-lr*b1_grad
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])

        if (epoch % 50 == 0):
            #每50个epoch，打印loss信息
            print('epoch:{},loss:{}'.format(epoch,loss_all / 4))
        train_loss_results.append(loss_all / 4)
        loss_all = 0

        total_corret,total_number = 0,0
        for x_test,y_test in test_db:
            y = tf.matmul(x_test,w1)+b1
            y = tf.nn.softmax(y)
            predict = tf.argmax(y,axis=1)             #返回y中最大值的索引，即预测的分类
            predict = tf.cast(predict,dtype=y_test.dtype)
            correct = tf.cast(tf.equal(predict,y_test),dtype=tf.int32)
            correct = tf.reduce_sum(correct)          #将每个batch中的correct数加起来
            total_corret += int(correct)
            total_number += x_test.shape[0]           #x_test.shape[0]代表样本个数

        acc = total_corret / total_number
        test_acc.append(acc)
        if (epoch % 50 == 0):
            print('test_acc:',acc)
            print('------------------')


    #绘制loss曲线
    plt.title('loss function curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,1)
    plt.plot(train_loss_results,label='$loss$')
    plt.legend()

    #绘制acc曲线
    plt.title('acc curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.plot(test_acc,label='$accuracy$')
    plt.legend()

    plt.show()

    w = w1.numpy()
    b = b1.numpy()
    y = np.dot(xt, w) + b
    max_index, max_number = max(enumerate(y), key=operator.itemgetter(1))
    return max_index  #返回概率最大的下标


print("请依次输入要预测的鸢尾花花萼长度、花萼宽度、花瓣长度、花瓣宽度：")
a1, a2, a3, a4 = input().split()
xt = np.array([a1,a2,a3,a4],dtype=np.float32)
ans_num = poj(xt)
flower = ['setosa','versicolor','virginica']
print("预测结果是：", end='')
print(flower[ans_num])
