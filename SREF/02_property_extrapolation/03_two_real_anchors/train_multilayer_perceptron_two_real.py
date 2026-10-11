# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:41:36 2022
bp神经网络
@author: Administrator
"""
#%%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV    #网格搜索
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
import matplotlib.pyplot as plt   #画图
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
import matplotlib.pyplot as plt   #画图
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score #交叉验证
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import sklearn
from sklearn.metrics import r2_score
import math
from sklearn.metrics import mean_squared_error #MSE
#%%
#导入数据集
data = pd.read_csv('02_property_extrapolation/03_two_real_anchors/training_low_hardness_with_two_real_anchors_145.csv', encoding ='gb2312')

data.drop_duplicates(inplace=True)
data.index = range(data.shape[0])  #恢复打乱的索引

data = data.iloc[:, 1:]
data.drop_duplicates(inplace=True) #删除重复值
data.index = range(data.shape[0])  #恢复打乱的索引

#分出特征和标签
x = data.iloc[:,1:]
y = data.iloc[:,0]

# max_x = x.max()
# min_x = x.min()
# x = (x - min_x) / (max_x - min_x)

###分训练集和测试集######
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.2,random_state=420)
#%%
#####建模#######
from sklearn.neural_network import MLPRegressor
from skopt import BayesSearchCV
from sklearn.metrics import r2_score

bpnn = MLPRegressor()

# Define the parameter space for Bayesian optimization with MLPRegressor
mytuple =[]
j = 10
for j in range(30,80,10):
    for i in range(5,10):
        ituple = (j,i)
        mytuple.append(ituple)
param_space = { "hidden_layer_sizes": mytuple,
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'max_iter': [20, 40, 60],
               }
bpnn_Grid = GridSearchCV(bpnn , param_space  , n_jobs=6)
bpnn_Grid.fit(Xtrain,Ytrain)

reg = MLPRegressor(
    **bpnn_Grid.best_params_
                    ).fit(Xtrain,Ytrain)#多层感知机



#%%

yhat = reg.predict(Xtest)
yhat_train = reg.predict(Xtrain)


r2_test = reg.score(Xtest,Ytest)
r2_train = reg.score(Xtrain,Ytrain)

MSE_train = mean_squared_error(Ytrain,yhat_train)
MSE_test = mean_squared_error(Ytest,yhat)

#####评估指标----MSE#######
from sklearn.metrics import mean_squared_error as MSE
MSE(yhat,Ytest)

import sklearn
sorted(sklearn.metrics.get_scorer_names())
print("MSE=",cross_val_score(reg,x,y,cv=10,scoring="neg_mean_squared_error").mean())



import matplotlib.pyplot as plt
sorted(Ytest)
plt.plot(range(len(Ytest)),sorted(Ytest),c="black",label= "Data")
plt.plot(range(len(yhat)),sorted(yhat),c="red",label = "Predict")
plt.legend()
plt.title("Linear regression")
plt.show()


#%%
score2 = []
score2 = cross_val_score(reg,x,y,cv=10)
MSE_10kf = cross_val_score(reg,x,y, cv=10    
                ,scoring = "neg_mean_squared_error").mean()

print("MLPR调参后：")
print("训练集R2：",r2_train) 
print("训练集RMSE：",math.sqrt(MSE_train))
print("测试集R2：",r2_test) 
print("测试集RMSE：",math.sqrt(MSE_test))
print("10折交叉验证R2：",score2.mean())
print("10折交叉验证RMSE：",math.sqrt(-MSE_10kf))

#%%
print("============高点预测==========")
import pickle #python内置模块
#将模块写入到lrmodel.pickle文件中
with open('multilayer_perceptron_two_real_MLPRmodel.pickle', 'wb') as fw: 
    pickle.dump(reg, fw)

# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('multilayer_perceptron_two_real_MLPRmodel.pickle', 'rb') as fr: 
    new_cbr = pickle.load(fr)

data = pd.read_csv('02_property_extrapolation/03_two_real_anchors/test_high_hardness_without_two_anchors_128.csv', encoding ='gb2312')
data = data.iloc[:, 1:]

data.drop_duplicates(inplace=True)
data.index = range(data.shape[0])  #恢复打乱的索引


Xtest = data.iloc[:,1:]
Ytest = data.iloc[:,0]

# max_x = x.max()
# min_x = x.min()
# Xtest = (Xtest - min_x) / (max_x - min_x)


#让我们用new_lr.predict()来对X_test预测一下 
yhat_new = new_cbr.predict(Xtest)

# 创建一个 DataFrame 包含这两个数据
df = pd.DataFrame({'Real': Ytest,'Pred': yhat_new})

# 将 DataFrame 写入 CSV 文件
df.to_csv('multilayer_perceptron_two_real_output_high.csv', index=False)


import matplotlib.pyplot as plt

# Scatter plot for comparing real values and predictions
plt.scatter(Ytest, yhat_new, c="red", label="Predictions")
plt.plot([min(Ytest), max(Ytest)], [min(Ytest), max(Ytest)], '--', c="black", label="Ideal Prediction")
plt.legend()
plt.title("Scatter Plot: Real Values vs Predictions")
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.show()
rmse_high = np.sqrt(mean_squared_error(Ytest, yhat_new))
print("RMSE:", rmse_high)
from sklearn.metrics import r2_score
r2_high = r2_score(Ytest, yhat_new)
print("R²:", r2_high)
#%%
print("============低点预测==========")

# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('multilayer_perceptron_two_real_MLPRmodel.pickle', 'rb') as fr: 
    new_cbr = pickle.load(fr)

data = pd.read_csv('02_property_extrapolation/03_two_real_anchors/training_low_hardness_with_two_real_anchors_145.csv', encoding ='gb2312')
data = data.iloc[:, 1:]

data.drop_duplicates(inplace=True)
data.index = range(data.shape[0])  #恢复打乱的索引


Xtest = data.iloc[:,1:]
Ytest = data.iloc[:,0]

# max_x = x.max()
# min_x = x.min()
# Xtest = (Xtest - min_x) / (max_x - min_x)


#让我们用new_lr.predict()来对X_test预测一下 
yhat_new = new_cbr.predict(Xtest)

# 创建一个 DataFrame 包含这两个数据
df = pd.DataFrame({'Real': Ytest,'Pred': yhat_new})

# 将 DataFrame 写入 CSV 文件
df.to_csv('multilayer_perceptron_two_real_output_low.csv', index=False)


import matplotlib.pyplot as plt

# Scatter plot for comparing real values and predictions
plt.scatter(Ytest, yhat_new, c="red", label="Predictions")
plt.plot([min(Ytest), max(Ytest)], [min(Ytest), max(Ytest)], '--', c="black", label="Ideal Prediction")
plt.legend()
plt.title("Scatter Plot: Real Values vs Predictions")
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.show()
rmse_low = np.sqrt(mean_squared_error(Ytest, yhat_new))
print("RMSE:", rmse_low)
from sklearn.metrics import r2_score
r2_low = r2_score(Ytest, yhat_new)
print("R²:", r2_low)