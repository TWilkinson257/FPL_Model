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

# Reading raw data in 
fn = 'players_raw.csv'
dataset = read_csv(fn)
# Teams 
teams = ['Arsenal', 'Aston Villa', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
         'Leicester', 'Leeds', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle', 'Sheffield Utd',
         'Southampton', 'Spurs', 'West Brom', 'West Ham', 'Wolves']

#Test Case
result_h = 3
result_a = 2

team_h = 14
team_a = 1


# print(home['minutes'].head())
# Definition of expected goals or assist
def xStat(team_val, Stat, result):
    # Finds team
    team = dataset[dataset['team'].values == team_val]
    # Expected xStat per 90, chance_of_playing * prob to be added
    xStat = team[Stat].values * 90 / team['minutes'].values
    # Normalising prob of xStat per 90
    xStat = np.nan_to_num(xStat)
    xStat_norm = xStat / sum(xStat)
    # Scaling normalised xStat to expected result
    xStat_GW = np.around(xStat_norm * result)
    return team['second_name'][xStat_GW > 0.5]

h_xG = xStat(team_h,'goals_scored',result_h)
h_xA = xStat(team_h,'assists',result_h)

a_xG = xStat(team_a,'goals_scored',result_a)
a_xA = xStat(team_a,'assists',result_a)

print('%s %.0f : %.0f %s' %(teams[team_h-1], result_h, result_a, teams[team_a-1]))
print('Goals:', h_xG.values, a_xG.values)
print('Asists:', h_xA.values, a_xA.values)
print()

# print(xG_GW)
# print(sum(xG_GW))


