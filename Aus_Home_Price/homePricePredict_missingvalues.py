#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:46:22 2020

@author: dhavaldangaria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:37:41 2020

@author: dhavaldangaria
"""


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#saving file path to variable
file_path='/Users/dhavaldangaria/Documents/Dhaval/kaggle/Home_price/train.csv'
file_path1='/Users/dhavaldangaria/Documents/Dhaval/kaggle/Home_price/test.csv'

X_train=pd.read_csv(file_path)
X_valid=pd.read_csv(file_path1)

print("----",X_train.columns) 
y=X_train['SalePrice']



train_X, val_X, train_y, val_y=train_test_split(X_train, y, random_state=1)

cols_with_missing=[col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print(cols_with_missing)
print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, train_y, val_y))




