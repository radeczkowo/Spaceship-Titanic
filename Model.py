import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from Functions import *

submission_df = pd.read_csv("Data/sample_submission.csv", index_col=0).drop(columns=['Transported'])
df_test = pd.read_csv('Data/df_test', index_col=0)
df_train = pd.read_csv('Data/df_train', index_col=0)
print(df_train.info())
print(df_test.info())
print(df_train.head())
df_train = sklearn.utils.shuffle(df_train)

# RidgeClassifier

print('RidgeClassifier:')
model = RidgeClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# RandomForestClassifier

print('RandomForestClassifier:')
model = RandomForestClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# ExtraTreesClassifier

print('ExtraTreesClassifier:')
model = ExtraTreesClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# DecisionTreeClassifier

print('DecisionTreeClassifier:')
model = DecisionTreeClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# AdaBoostClassifier

print('AdaBoostClassifier:')
model = AdaBoostClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# SGDClassifier

print('SGDClassifier:')
model = SGDClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# PassiveAggressiveClassifier

print('PassiveAggressiveClassifier:')
model = PassiveAggressiveClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# SVC

print('SVC:')
model = svm.SVC()
testmodel(df_train, model, 'Transported', 0.1)

# NuSVC

print('NuSVC:')
model = svm.NuSVC()
testmodel(df_train, model, 'Transported', 0.1)

# LinearSVC

print('LinearSVC:')
model = svm.LinearSVC()
testmodel(df_train, model, 'Transported', 0.1)

# Final model
model = svm.SVC()
finalprediction(model, df_train, df_test, 'Transported', submission_df)