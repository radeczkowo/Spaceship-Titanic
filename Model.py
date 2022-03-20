import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from Functions import *

submission_df = pd.read_csv("Data/sample_submission.csv", index_col=0).drop(columns=['Transported'])
df_test = pd.read_csv('Data/df_test', index_col=0)
df_train = pd.read_csv('Data/df_train', index_col=0)
print(df_train.info())
print(df_test.info())
print(df_train.head())
df_train = sklearn.utils.shuffle(df_train)

# RidgeClassifier

model = RidgeClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# RandomForestClassifier

model = RandomForestClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# ExtraTreesClassifier

model = ExtraTreesClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# DecisionTreeClassifier

model = DecisionTreeClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# AdaBoostClassifier

model = AdaBoostClassifier()
testmodel(df_train, model, 'Transported', 0.1)

# Final model
model = RandomForestClassifier()
finalprediction(model, df_train, df_test, 'Transported', submission_df)