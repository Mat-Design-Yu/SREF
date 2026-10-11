# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:43:34 2022
用XGB预测粘度，数据集为包含所有特征的  KongJianChaZhiTest.csv

@author: Administrator
"""
#导入库
from xgboost import XGBRegressor as XGBR
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
data = pd.read_csv('02_property_extrapolation/01_midrange_comparison/training_hardness_200_to_800_190.csv', encoding ='gb2312')

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


# Define the parameter space for Bayesian optimization with more aggressive ranges
param_space = {
    'n_estimators': (50, 500),  # Increase the range for the number of trees
    'learning_rate': (0.001, 0.5),  # Increase the range for the learning rate
    'subsample': (0.1, 1.0),  # Increase the range for the subsample ratio of the training instances
    'gamma': (0, 10),  # Increase the range for the minimum loss reduction required to make a further partition on a leaf node of the tree
    'max_depth': (3, 20),  # Increase the range for the maximum depth of a tree
    'min_child_weight': (1, 20)  # Increase the range for the minimum sum of instance weight needed in a child
}


# Create an XGBRegressor instance
xgb_reg = XGBR(random_state=420, booster="gbtree")

# Define the BayesSearchCV with the XGBRegressor and the parameter space
bayes_search = BayesSearchCV(
    xgb_reg,
    param_space,
    n_iter=50,  # Number of parameter settings that are sampled
    scoring='neg_mean_squared_error',  # Scoring metric (negative MSE)
    cv=5,
    n_jobs=-1
)

# Perform Bayesian optimization
bayes_search.fit(Xtrain, Ytrain)

# Print the best parameters and the best score
print(bayes_search.best_params_)
print(bayes_search.best_score_)

reg = bayes_search.best_estimator_


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

print("XGBR调参后：")
print("训练集R2：",r2_train) 
print("训练集RMSE：",math.sqrt(MSE_train))
print("测试集R2：",r2_test) 
print("测试集RMSE：",math.sqrt(MSE_test))
print("10折交叉验证R2：",score2.mean())
print("10折交叉验证RMSE：",math.sqrt(-MSE_10kf))

#%%
print("============中间预测==========")
import pickle #python内置模块
#将模块写入到lrmodel.pickle文件中
with open('xgboost_midrange_XGBRmodel.pickle', 'wb') as fw: 
    pickle.dump(reg, fw)


# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('xgboost_midrange_XGBRmodel.pickle', 'rb') as fr: 
    new_cbr = pickle.load(fr)

data = pd.read_csv('02_property_extrapolation/01_midrange_comparison/training_hardness_200_to_800_190.csv', encoding ='gb2312')
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
df.to_csv('xgboost_midrange_output_middle.csv', index=False)


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
print("============高点预测==========")
# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('xgboost_midrange_XGBRmodel.pickle', 'rb') as fr: 
    new_cbr = pickle.load(fr)

data = pd.read_csv('02_property_extrapolation/01_midrange_comparison/upper_extrapolation_hardness_above_800_29.csv', encoding ='gb2312')
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
df.to_csv('xgboost_midrange_output_high.csv', index=False)


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
print("============低点预测==========")

# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('xgboost_midrange_XGBRmodel.pickle', 'rb') as fr: 
    new_cbr = pickle.load(fr)

data = pd.read_csv('02_property_extrapolation/01_midrange_comparison/lower_extrapolation_hardness_at_most_200_55.csv', encoding ='gb2312')
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
df.to_csv('xgboost_midrange_output_low.csv', index=False)


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