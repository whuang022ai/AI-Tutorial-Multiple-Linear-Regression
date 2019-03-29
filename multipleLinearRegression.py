# coding=UTF-8
# 多元線性回歸 (使用梯度下降法)
import numpy as np
import pandas as pd
weights = np.zeros(3)
learing_rate = 0.0002 # 學習速率=0.0002
def multiple_liner_regression(X, Y, weights, learing_rate,epoch):
    n = len(Y)
    for i in range(epoch):
        # 計算Y_Hat-Y
        error=  X.dot(weights) - Y
        # 計算sum ( (Y_Hat-Y)*Xm)
        gradient = X.T.dot(error) / n
        # 新權重=舊權重-學習速率*梯度
        weights = weights - learing_rate * gradient
        cost=0
        for e in range(len(error)):
            cost+=0.5*error[e]**2
        # print(cost/n)
    return weights
data = pd.read_csv('stu.csv') # 利用pandas 讀取數據
math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values
m = len(math)
# 偏全值等價於X固定=1 bias=1,x1=math,x2= read
X = np.array([np.ones(m), math, read]).T
Y = np.array(write)
# 求出最佳權重
weights= multiple_liner_regression(X, Y, weights, learing_rate, 500000)
print('The bias and weights are:')
print(weights)
print('If math=48 and reading=68 than writing=' )
predict=weights[0]+weights[1]*48+weights[2]*68# 簡易測試
print(predict)
print('real data = 63' )
