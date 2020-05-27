#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:56:23 2020

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

#print the data
#print(mel_data.describe())

print(mel_data.columns)
#print(mel_data.Price)

mel_features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X=mel_data[mel_features]

y=mel_data['Price']

#print(X.describe())
#mel_model=DecisionTreeRegressor(random_state=1)
#mel_model.fit(X,y)

train_X, val_X, train_y, val_y=train_test_split(X, y, random_state=1)

mel_model=DecisionTreeRegressor()
mel_model.fit(train_X, train_y)

#print("Making predictions for the following 5 houses:")
#print(X.head())
#print("The predictions are")
#/print(mel_model.predict(X.head()))
#print(mean_absolute_error(y,mel_model.predict(X)))

val_predictions=mel_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


