import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 200)

df_all = pd.read_csv('Data/df_all_V2', index_col=0)


# Cabin, Group - Dropped due to too big diversity

df_all.drop(columns=['Cabin', 'CabinNr', 'Group'], inplace=True)

# DestHome

df_all['DestHome'] = df_all['Destination'] + df_all['HomePlanet']
group = df_all.groupby(['DestHome'])['Transported'].mean().sort_values(ascending=False)
group.plot(kind='bar')
print(group)
# plt.show()
mapping = {'TRAPPIST-1eEarth': 1, 'PSO J318.5-22Mars': 2, 'PSO J318.5-22Earth': 3, '55 Cancri eEarth': 4,
           'TRAPPIST-1eMars': 5, '55 Cancri eMars': 6, 'TRAPPIST-1eEuropa': 7, '55 Cancri eEuropa': 8,
           'PSO J318.5-22Europa': 9}
df_all['DestHome'] = df_all['DestHome'].map(mapping).astype(int)

# HomePlanet

group = df_all.groupby(['HomePlanet'])['Transported'].mean().sort_values(ascending=False)
group.plot(kind='bar')
print(group)
# plt.show()
mapping = {'Earth': 1, 'Mars': 2, 'Europa': 3}
df_all['HomePlanet'] = df_all['HomePlanet'].map(mapping).astype(int)

# Destination

group = df_all.groupby(['Destination'])['Transported'].mean().sort_values(ascending=False)
group.plot(kind='bar')
print(group)
# plt.show()
mapping = {'55 Cancri e': 1, 'PSO J318.5-22': 2, 'TRAPPIST-1e': 3}
df_all['Destination'] = df_all['Destination'].map(mapping).astype(int)

# CryoSleep

group = df_all.groupby(['CryoSleep'])['Transported'].mean().sort_values(ascending=False)
group.plot(kind='bar')
print(group)
# plt.show()
mapping = {False: 1, True: 2}
df_all['CryoSleep'] = df_all['CryoSleep'].map(mapping).astype(int)

# DestHomeSleep

df_all['DestHomeSleep'] = df_all['DestHome'] * df_all['CryoSleep']
group = df_all.groupby(['DestHomeSleep'])['Transported'].mean().sort_values(ascending=False)
group.plot(kind='bar')
print(group)
# plt.show()

# mapping = {'TRAPPIST-1eMars': 1, 'TRAPPIST-1eEarth': 2, 'PSO J318.5-22Mars': 3, 'PSO J318.5-22Earth': 4,
#            '55 Cancri eMars': 5, 'Cancri eEuropa': 6, '55 Cancri eEarth': 7, 'TRAPPIST-1eEuropa': 8,
#            'PSO J318.5-22Europa': 9, 'TRAPPIST-1eEarthTRAPPIST-1eEarth': 10, 'PSO J318.5-22EarthPSO J318.5-22Earth': 11,
#            '55 Cancri eEarth55 Cancri eEarth': 12, 'PSO J318.5-22MarsPSO J318.5-22Mars': 13,
#            'TRAPPIST-1eMarsTRAPPIST-1eMars': 14, '55 Cancri eMars55 Cancri eMars': 15,
#            'TRAPPIST-1eEuropaTRAPPIST-1eEuropa': 16, '55 Cancri eEuropa55 Cancri eEuropa': 17,
#            'PSO J318.5-22EuropaPSO J318.5-22Europa': 18}
# df_all['DestHomeSleep'] = df_all['DestHomeSleep'].map(mapping).astype(int)

# Age

df_all["AgeBinds"] = pd.cut(df_all["Age"], [-1, 5, 10, 20, 30, 1000], labels=["0-5", "5-10", "10-20", "20-30", "30+"])
group = df_all.groupby(['AgeBinds'])['Transported'].mean().sort_values(ascending=False)
group.plot(kind='bar')
print(group)
# plt.show()

mapping = {"30+": 1, "20-30": 2, "10-20": 3, "5-10": 4, "0-5": 5}
df_all['AgeBinds'] = df_all['AgeBinds'].map(mapping).astype(int)

# VIP

group = df_all.groupby(['VIP'])['Transported'].mean().sort_values(ascending=False)
group.plot(kind='bar')
print(group)
# plt.show()

mapping = {False: 2, True: 1}
df_all['VIP'] = df_all['VIP'].map(mapping).astype(int)

# GroupCount

group = df_all.groupby(['GroupCount'])['Transported'].mean().sort_values(ascending=False)
# plt.clf()
group.plot(kind='bar')
print(group)
# plt.show()

mapping = {8: 1, 1: 2, 2: 3, 7: 4, 5: 5, 3: 6, 6: 7, 4: 8}
df_all['GroupCount'] = df_all['GroupCount'].map(mapping).astype(int)

# CabinA

group = df_all.groupby(['CabinA'])['Transported'].mean().sort_values(ascending=False)
plt.clf()
group.plot(kind='bar')
print(group)
# plt.show()

mapping = {"T": 1, "E": 2, "D": 3, "F": 4, "None": 5, "A": 6, "G": 7, "C": 8, "B": 9}
df_all['CabinA'] = df_all['CabinA'].map(mapping).astype(int)

# CabinPS

group = df_all.groupby(['CabinPS'])['Transported'].mean().sort_values(ascending=False)
plt.clf()
group.plot(kind='bar')
print(group)
# plt.show()
mapping = {'None': 1, 'P': 2, "S": 3}
df_all['CabinPS'] = df_all['CabinPS'].map(mapping).astype(int)

# SurnameGroupSize

group = df_all.groupby(['SurnameGroupSize'])['Transported'].mean().sort_values(ascending=False)
plt.clf()
group.plot(kind='bar')
print(group)
# plt.show()

mapping = {8: 1, 7: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
df_all['SurnameGroupSize'] = df_all['SurnameGroupSize'].map(mapping).astype(int)

# CabinNrBind

group = df_all.groupby(['CabinNrBind'])['Transported'].mean().sort_values(ascending=False)
plt.clf()
group.plot(kind='bar')
print(group)
# plt.show()
print(df_all.info())

mapping = {"G": 1, "F": 2, "B": 3, "None": 4, "C": 5, "A": 6, "E": 7, "D": 8}
df_all['CabinNrBind'] = df_all['CabinNrBind'].map(mapping).astype(int)


# Dropping unnecessary needed columns

df_all.drop(columns=['PassengerId', 'Surname', 'DestHomeSleep'], inplace=True)

print(df_all.info())
# print(df_all.corr())
# print(df_all.describe())

df_all.to_csv('Data/df_all_V3', index=True)