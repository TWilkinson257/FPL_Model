#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:41:12 2020

@author: thomaswilkinson
"""

# load libraries
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
from sklearn.svm import SVC
import csv
from itertools import permutations
from math import factorial
import numpy as np
# from itertools import product
pd.options.display.max_columns = None


# Reading raw data in 
fn = 'players_raw.csv'
dataset = read_csv(fn)
# Statistically describe dataset
print(dataset.describe())

# create arrays to define whether entries count against a limit
costs = dataset['now_cost']
gk = np.array(dataset['element_type'].values == 1)
defend = np.array(dataset['element_type'].values == 2)
mid = np.array(dataset['element_type'].values == 3)
att = np.array(dataset['element_type'].values == 4)
ast = np.array(dataset['team'].values == 2) # set to villa currently
#  i.e every player counts against player limit
players = np.ones(len(dataset))

# Set up boundaries
params = np.array([
    costs, 
    gk,
    defend,
    mid,
    att,
    ast,
    players,
   ])

# Set limits
upper_bounds = np.array([
    970, #cost
    2, #gk
    5, #def
    5, #mid
    3, #att
    3, #players from team
    15, #player limit
    ])

total_points = dataset['total_points']

bounds = [(0,1) for x in range(len(dataset))]

# linear progression to solve minimise negative total points (so most points at end)
from scipy.optimize import linprog

selected = linprog(
    -total_points, 
    params, 
    upper_bounds,
    bounds = bounds,
   ).x

#round to tidy answers
selected_rnd = np.around(selected)

gk_selected = gk * selected_rnd
def_selected = defend * selected_rnd
mid_selected = mid * selected_rnd
att_selected = att * selected_rnd

print('cost: %.2f' % (dataset['now_cost'] * selected_rnd).sum())
print('Gk: %.2f' % ( gk_selected).sum())
print('Def: %.2f' % (def_selected).sum())
print('Mid: %.2f' % (mid_selected).sum())
print('Att: %.2f' % (att_selected).sum())

print("Goalkeepers:","\n",dataset['web_name'][gk_selected == 1],
      "Defenders:", "\n", dataset['web_name'][def_selected == 1],
      "Midfielders:", "\n", dataset['web_name'][mid_selected == 1],
      "Strikers:", "\n", dataset['web_name'][att_selected == 1])


