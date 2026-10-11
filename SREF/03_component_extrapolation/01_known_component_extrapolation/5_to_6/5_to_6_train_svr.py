# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:26:13 2022
支持向量机回归
@author: Administrator
"""
#%%####导入所需要的库#############
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
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
data = pd.read_csv('03_component_extrapolation/01_known_component_extrapolation/data/quinary_hardness_compositions.csv', encoding ='gb2312')

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
reg = svm.SVR()
reg.fit(Xtrain,Ytrain)

# Define hyperparameter search space
search_spaces = {
    # 'kernel': Categorical(['poly', 'rbf', 'sigmoid']),
    'kernel': Categorical([ 'rbf']),
    'C': Real(10, 100, prior='log-uniform'),
    'gamma': Real(1e-6, 10, prior='log-uniform'),  # Adjusted upper bound for gamma
    'degree': Integer(4, 10)
}

# Use Bayesian Optimization for hyperparameter search
opt = BayesSearchCV(
    estimator=svm.SVR(cache_size=2000),
    search_spaces=search_spaces,
    n_iter=15,  # Reduced number of iterations for faster computation
    scoring="neg_mean_squared_error",  # Evaluation metric
    cv=3,  # Reduced number of folds for cross-validation
    n_jobs=-1  # Utilize all available CPU cores
)

# 运行优化
opt.fit(Xtrain, Ytrain)

# 输出最佳参数
print("Best parameters found: ", opt.best_params_)
print("Cross-validation R^2 score with best parameters: ", opt.best_score_)

# 使用最佳参数的模型进行预测
best_model = opt.best_estimator_

reg = best_model
yhat = reg.predict(Xtest)
r2 = r2_score(yhat, Ytest)
print("Test R^2 score with best parameters: ", r2)


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

print("SVM调参后：")
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
with open('5_to_6_train_svr_SVMmodel.pickle', 'wb') as fw: 
    pickle.dump(reg, fw)

# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('5_to_6_train_svr_SVMmodel.pickle', 'rb') as fr: 
    new_cbr = pickle.load(fr)

data = pd.read_csv('03_component_extrapolation/01_known_component_extrapolation/data/senary_hardness_compositions.csv', encoding ='gb2312')
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
df.to_csv('5_to_6_train_svr_output_6D.csv', index=False)


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

# 通过pickle.load 加载 lrmodel.pickle文件，把读取出来的模型赋予一个新的对象new_lr
with open('5_to_6_train_svr_SVMmodel.pickle', 'rb') as fr: 
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
df.to_csv('5_to_6_train_svr_output_low.csv', index=False)


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