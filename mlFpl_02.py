# import requests
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

# Reading fixture data in
fn = 'fixtures.csv'
fixtures = read_csv(fn)

fixtures_h = fixtures[['team_a_difficulty','team_h_difficulty', 'team_h_score' ]][fixtures['minutes'].values == 90]
fixtures_a = fixtures[['team_a_difficulty','team_h_difficulty', 'team_a_score' ]][fixtures['minutes'].values == 90]

# Reading raw player data in 
fn = 'players_raw.csv'
players_raw = read_csv(fn)

# print(fixtures_h.head(10))
# print(fixtures_a.head(10))

teams = ['Arsenal', 'Aston Villa', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
         'Leicester', 'Leeds', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle', 'Sheffield Utd',
         'Southampton', 'Spurs', 'West Brom', 'West Ham', 'Wolves']

# Predicting scores from previous FDR results 

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

#model definition
model_a = SVR()
model_a.fit(X, y)

# test_fix = [4, 2]
# test_h_score = model_h.predict([test_fix])
# test_a_score = model_a.predict([test_fix])

# print('%.0f : %.0f' %(test_h_score, test_a_score))
# Prediction of scores        
pred_h_score = np.array([])
pred_a_score = np.array([])

for row in range(len(fixtures)):
    h_diff = fixtures['team_h_difficulty'][row]
    a_diff = fixtures['team_a_difficulty'][row]
    pred_h_score = np.append(pred_h_score,model_h.predict([[h_diff,a_diff]]))
    pred_a_score = np.append(pred_a_score,model_a.predict([[h_diff,a_diff]]))
    
    # pred_h_score.append(model_h.predict([[h_diff,a_diff]]))
    # pred_a_score.append(model_a.predict([[h_diff,a_diff]]))

# Definition of expected goals or assist for given team and result
def xStat(team_val, Stat, result):
    # Finds team
    team = players_raw[players_raw['team'].values == team_val]
    # Expected xStat per 90, chance_of_playing * prob to be added
    xStat_temp = team[Stat].values * 90 / team['minutes'].values
    # Normalising prob of xStat per 90
    xStat_temp = np.nan_to_num(xStat_temp)
    xStat_norm = xStat_temp / sum(xStat_temp)
    xStat_sort = np.sort(xStat_norm)
    xStat_plyrsort = np.argsort(xStat_norm)
    # print(team['second_name'].values[xStat_plyrsort])
    xStat_cum = np.cumsum(xStat_sort)
    # xStat_GW = [[np.zeros(len(xStat_cum))]]
    xStat_GW = []
    # print(xStat_cum)
    for event in range(result):    
        seed = np.random.rand(1)
        # print(seed)
        for plyr in range(len(xStat_cum)):
            if xStat_cum[plyr] > seed:
                plyr_stat = xStat_plyrsort[plyr]
                xStat_GW.append(team['second_name'].values[plyr_stat])
                break
    # xStat_GW = np.around(xStat_norm * result)
    # return team['second_name'][xStat_GW == 1]
    return xStat_GW

#Printing predictions to terminal 
current_gw = 9
gw_fixt = fixtures[(fixtures['event'].values == current_gw)]

for row in range(len(pred_h_score[fixtures['event'].values == current_gw])):
    # Define teams and result for fixture
    teamcode_h = gw_fixt['team_h'].values[row]
    teamcode_a = gw_fixt['team_a'].values[row]
    team_h = teams[teamcode_h - 1]
    team_a = teams[teamcode_a - 1]
    result_h = int(round(pred_h_score[fixtures['event'].values == current_gw][row]))
    result_a = int(round(pred_a_score[fixtures['event'].values == current_gw][row]))

    #Calculate contributions
    h_xG = xStat(teamcode_h,'goals_scored',result_h)
    h_xA = xStat(teamcode_h,'assists',result_h)
    a_xG = xStat(teamcode_a,'goals_scored',result_a)
    a_xA = xStat(teamcode_a,'assists',result_a)

    print('%s %.0f : %.0f %s' %(team_h, result_h, result_a, team_a))
    print('Goals:', h_xG, a_xG)
    print('Asists:', h_xA, a_xA)
    print() 
    # print(pred_a_score[fixtures['event'].values == 7][row])

