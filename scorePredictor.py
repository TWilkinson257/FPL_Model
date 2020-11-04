#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:44:11 2020

@author: thomaswilkinson
"""

import requests
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
import csv
from itertools import permutations
from math import factorial
import numpy as np

fn = 'fixtures.csv'
fixtures = read_csv(fn)

fixtures_h = fixtures[['team_a_difficulty','team_h_difficulty', 'team_h_score' ]][fixtures['minutes'].values == 90]
fixtures_a = fixtures[['team_a_difficulty','team_h_difficulty', 'team_a_score' ]][fixtures['minutes'].values == 90]

# print(fixtures_h.head(10))
# print(fixtures_a.head(10))

teams = ['Arsenal', 'Aston Villa', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
         'Leicester', 'Leeds', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle', 'Sheffield Utd',
         'Southampton', 'Spurs', 'West Brom', 'West Ham', 'Wolves']

# Split-out validation dataset - home scores
array = fixtures_h.values
X = array[:,0:2]
y = array[:,2]

# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

model_h = SVR()
model_h.fit(X, y)

# kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
# cv_results = cross_val_score(model_h, X_train, Y_train, cv=kfold, scoring='accuracy')


# Split-out validation dataset - away scores
array = fixtures_a.values
X = array[:,0:2]
y = array[:,2]

model_a = SVR()
model_a.fit(X, y)

# test_fix = [4, 2]
# test_h_score = model_h.predict([test_fix])
# test_a_score = model_a.predict([test_fix])

# print('%.0f : %.0f' %(test_h_score, test_a_score))
        
pred_h_score = np.array([])
pred_a_score = np.array([])

for row in range(len(fixtures)):
    h_diff = fixtures['team_h_difficulty'][row]
    a_diff = fixtures['team_a_difficulty'][row]
    pred_h_score = np.append(pred_h_score,model_h.predict([[h_diff,a_diff]]))
    pred_a_score = np.append(pred_a_score,model_a.predict([[h_diff,a_diff]]))
    
    # pred_h_score.append(model_h.predict([[h_diff,a_diff]]))
    # pred_a_score.append(model_a.predict([[h_diff,a_diff]]))
   
current_gw = 7
gw_fixt = fixtures[(fixtures['event'].values == current_gw)]

for row in range(len(pred_h_score[fixtures['event'].values == current_gw])):
    # print(teams[gw_fixt['team_h'].values[row]-1])
    # print(teams[gw_fixt['team_a'].values[row]-1])
    print('%s %.0f : %.0f %s' %(teams[gw_fixt['team_h'].values[row]-1], pred_h_score[fixtures['event'].values == current_gw][row], 
          pred_a_score[fixtures['event'].values == current_gw][row], teams[gw_fixt['team_a'].values[row]-1]))
    # print(pred_a_score[fixtures['event'].values == 7][row])

