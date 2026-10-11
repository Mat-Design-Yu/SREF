# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:03:54 2022

@author: Administrator
"""
###导入模块和库####
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV    #网格搜索
from sklearn.model_selection import cross_val_score #交叉验证
import matplotlib.pyplot as plt   #画图
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error #MSE

#%%
#导入数据集
data = pd.read_csv('03_component_extrapolation/01_known_component_extrapolation/data/quaternary_hardness_compositions.csv', encoding ='gb2312')

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
# # Linear Regression
reg = LinearRegression().fit(Xtrain, Ytrain)


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
sorted(sklearn.metrics.SCORERS.keys())
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

print("LR调参后：")
print("训练集R2：",r2_train) 
print("训练集RMSE：",math.sqrt(MSE_train))
print("测试集R2：",r2_test) 
print("测试集RMSE：",math.sqrt(MSE_test))
print("10折交叉验证R2：",score2.mean())
print("10折交叉验证RMSE：",math.sqrt(-MSE_10kf))

#%%
print("============高组元预测==========")
import pickle #python内置模块
#将模块写入到lrmodel.pickle文件中
with open('4_to_5_train_linear_lrmodel.pickle', 'wb') as fw: 
    pickle.dump(reg, fw)


# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('4_to_5_train_linear_lrmodel.pickle', 'rb') as fr: 
    new_cbr = pickle.load(fr)

data = pd.read_csv('03_component_extrapolation/01_known_component_extrapolation/data/quinary_hardness_compositions.csv', encoding ='gb2312')
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
df.to_csv('4_to_5_train_linear_output_5D.csv', index=False)


import matplotlib.pyplot as plt

# Scatter plot for comparing real values and predictions
plt.scatter(Ytest, yhat_new, c="red", label="Predictions")
plt.plot([min(Ytest), max(Ytest)], [min(Ytest), max(Ytest)], '--', c="black", label="Ideal Prediction")
plt.legend()
plt.title("Scatter Plot: Real Values vs Predictions")
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.show()
rmse = np.sqrt(mean_squared_error(Ytest, yhat_new))
print("RMSE:", rmse)
from sklearn.metrics import r2_score
r2 = r2_score(Ytest, yhat_new)
print("R²:", r2)
#%%
print("============低组元预测==========")
import pickle #python内置模块
#将模块写入到lrmodel.pickle文件中
with open('4_to_5_train_linear_lrmodel.pickle', 'wb') as fw: 
    pickle.dump(reg, fw)


# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('4_to_5_train_linear_lrmodel.pickle', 'rb') as fr: 
    new_cbr = pickle.load(fr)

data = pd.read_csv('03_component_extrapolation/01_known_component_extrapolation/data/quaternary_hardness_compositions.csv', encoding ='gb2312')
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
df.to_csv('4_to_5_train_linear_output_low.csv', index=False)


import matplotlib.pyplot as plt

# Scatter plot for comparing real values and predictions
plt.scatter(Ytest, yhat_new, c="red", label="Predictions")
plt.plot([min(Ytest), max(Ytest)], [min(Ytest), max(Ytest)], '--', c="black", label="Ideal Prediction")
plt.legend()
plt.title("Scatter Plot: Real Values vs Predictions")
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.show()
rmse = np.sqrt(mean_squared_error(Ytest, yhat_new))
print("RMSE:", rmse)
from sklearn.metrics import r2_score
r2 = r2_score(Ytest, yhat_new)
print("R²:", r2)