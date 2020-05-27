#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:21:20 2020

@author: dhavaldangaria
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


#saving file path to variable
mel_file_path='/Users/dhavaldangaria/Documents/Dhaval/kaggle/Aus_Home_Price/melb_data.csv'

#read teh data and store in DataFrame
mel_data=pd.read_csv(mel_file_path)

mel_features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X=mel_data[mel_features]

y=mel_data['Price']
train_X, val_X, train_y, val_y=train_test_split(X, y, random_state=1)

def get_mae(max_leaf_nodes,train_X, train_y, val_X, val_y):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val= model.predict(val_X)
    mae=mean_absolute_error(val_y, preds_val)

    return mae

for i in [5,50,500,5000]:
    my_mae=get_mae(i, train_X, train_y, val_X, val_y)
    print("Max leaf nodes: %d \t\t mean abolute error: %d" %(i, my_mae))
    

