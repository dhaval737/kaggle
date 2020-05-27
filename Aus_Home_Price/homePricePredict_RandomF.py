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


#saving file path to variable
mel_file_path='/Users/dhavaldangaria/Documents/Dhaval/kaggle/Aus_Home_Price/melb_data.csv'

#read teh data and store in DataFrame
mel_data=pd.read_csv(mel_file_path)

mel_features=[c]

X=mel_data[mel_features]

y=mel_data['Price']
train_X, val_X, train_y, val_y=train_test_split(X, y, random_state=1)

forest_model=RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

mel_preds=forest_model.predict(val_X)
print(mean_absolute_error(val_y, mel_preds))

