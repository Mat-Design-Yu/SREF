# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:46:06 2023
通过公式计算特征
@author: 25304
"""


import numpy as np
import pandas as pd
from sympy import *
from pysr import PySRRegressor

data = pd.read_csv('02_property_extrapolation/06_composition_design/design_candidate_compositions_500000.csv', encoding ='gb2312')
data = data.iloc[:, 1:]


# 读取第一个数据集
df1 = data


# 读取CSV文件
equations = pd.read_csv('02_property_extrapolation/06_composition_design/four_ensemble_symbolic_equations.csv')

# 读取第二个数据集
df2 = pd.DataFrame(equations["sympy_format"])




# Mapping column names to symbols for sympy
var_map = {col: Symbol(col) for col in df1.columns}

# Convert sympy formula to lambda functions and store them in df2
for idx, row in df2.iterrows():
    formula = lambdify(var_map.keys(), sympify(row['sympy_format']))
    df2.at[idx, 'formula'] = formula

# Compute the formulas and add results to df1
for idx, row in df1.iterrows():
    for idx2, row2 in df2.iterrows():
        formula = row2['formula']
        if callable(formula):
            try:
                # Ensure inputs are safe for computation, e.g., checking for sin(0) or other potential issues
                result = formula(*row)
                
                # Handle specific cases like sin(0), log(0) if required
                if np.isnan(result) or np.isinf(result):
                    result = None  # Assign None if the result is invalid
                # Store the result in df1
                df1.at[idx, f"result_{idx2}"] = result
            except Exception as e:
                # In case of error (e.g., invalid computation), assign None
                df1.at[idx, f"result_{idx2}"] = None
        else:
            break
    else:
        continue
    break


# 将结果保存到新的Excel文件中
df1.to_csv('design_candidate_equations_CD-Addequations.csv')