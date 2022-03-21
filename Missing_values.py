import pandas as pd

from Functions import *
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 200)

df_all = pd.read_csv('Data/df_all_V0', index_col=0)

print(df_all.info())
print(df_all.corr())
print(df_all.describe())

# HomePlanet - People in the same group and cabin are more likely from the same planet

g_hom_dest = df_all.groupby(['Destination', 'HomePlanet'])['PassengerId'].count().unstack()
print(g_hom_dest)
g_hom_dest.plot(kind='bar')

g_hom_dest = df_all.groupby(['CryoSleep', 'HomePlanet'])['PassengerId'].count().unstack()
print(g_hom_dest)
g_hom_dest.plot(kind='bar')
# plt.show()


df_all = fillmissingvalues(df_all, ['Destination', 'CryoSleep', 'Cabin', 'Group'], 'HomePlanet', [1.0, 1.0, 3.0, 2.0])


# CryoSleep - People who have vip did not go into hibernation in most cases

g_hom_dest = df_all.groupby(['Destination', 'CryoSleep'])['PassengerId'].count().unstack()
print(g_hom_dest)
g_hom_dest.plot(kind='bar')
g_hom_dest = df_all.groupby(['HomePlanet', 'CryoSleep'])['PassengerId'].count().unstack()
print(g_hom_dest)
g_hom_dest.plot(kind='bar')
g_hom_dest = df_all.groupby(['VIP', 'CryoSleep'])['PassengerId'].count().unstack()
print(g_hom_dest)
g_hom_dest.plot(kind='bar')
# plt.show()

df_all = fillmissingvalues(df_all, ['VIP', 'HomePlanet', 'Destination'], 'CryoSleep',
                           [1.4, 1, 1])


# Destination - People in the same group and cabin are more likely go the same planet

g_hom_dest = df_all.groupby(['Destination', 'HomePlanet'])['PassengerId'].count().unstack()
print(g_hom_dest)
g_hom_dest.plot(kind='bar')

g_hom_dest = df_all.groupby(['Destination', 'CryoSleep'])['PassengerId'].count().unstack()
print(g_hom_dest)
g_hom_dest.plot(kind='bar')
# plt.show()

df_all = fillmissingvalues(df_all, ['HomePlanet', 'CryoSleep', 'Cabin', 'Group'], 'Destination', [1.0, 1.0, 1.5, 2.0])


# Cabin - Decided to put people with the same surname and group in one cabin together where nulls occur. Rest of
# replacement for null none

print(df_all.info())
w = df_all.groupby(['Group', 'Surname', 'Cabin'])['Surname'].count()
w = pd.DataFrame(w)
w.rename(columns={'Surname': 'SCount'}, inplace=True)

w.reset_index(inplace=True)
w.set_index("Group", inplace=True)

z = w.groupby(['Group', 'Surname'])['Cabin'].last()
z = pd.DataFrame(z)
z.reset_index(inplace=True)
z.set_index("Group", inplace=True)
z.rename(columns={'Surname': 'SurnameN', 'Cabin': "CabinN"}, inplace=True)

df_all = df_all.merge(z, on=['Group'], how='outer')
# print(df_all[['PassengerId', 'Cabin', 'Group', 'Surname', 'SurnameN', 'CabinN']].head(130))

df_nans = df_all.loc[df_all[['Cabin']].isnull().any(axis=1) & (df_all['Surname'] == df_all['SurnameN'])]
df_nans['Cabin'] = df_nans['CabinN']
df_all.loc[df_all[['Cabin']].isnull().any(axis=1) & (df_all['Surname'] == df_all['SurnameN'])] = df_nans
df_all.set_index('PassengerId', inplace=True)
df_all = df_all[~df_all.index.duplicated(keep='first')]

df_all.reset_index(inplace=True)

df_all['Cabin'].fillna('None/None/None', inplace=True)

df_all[["CabinA", 'CabinNr', 'CabinPS']] = df_all["Cabin"].str.split('/', expand=True)

print(df_all.info())

df_all['Surname'] = df_all['Surname'].fillna('Unknown')


# Age


df_all['Age'] = df_all['Age'].fillna(df_all.groupby('CabinA')['Age'].transform('mean'))

# Vip
g_hom_dest = df_all.groupby(['VIP', 'Destination'])['PassengerId'].count().unstack()
print(g_hom_dest)
g_hom_dest.plot(kind='bar')

df_all = fillmissingvalues(df_all, ['CryoSleep', 'Cabin', 'Group'], 'VIP', [1.2, 1.0, 1.0])


print(df_all.skew())

# RoomService

df_all.loc[(df_all['RoomService'] != 0), 'RoomService'] = np.log1p(df_all.loc[(df_all['RoomService'] != 0),
                                                                              'RoomService'])

nans = df_all["RoomService"].isna()
newvalues = np.random.randint(df_all["RoomService"].mean() - df_all["RoomService"].std(), df_all["RoomService"].mean()
                              + df_all["RoomService"].std(), len(nans))
newvalues = np.where(newvalues < 0, 0, newvalues)
df_all["RoomService"][np.isnan(df_all["RoomService"])] = newvalues

# FoodCourt Using linear regression

df_all.loc[(df_all['FoodCourt'] != 0), 'FoodCourt'] = np.log1p(df_all.loc[(df_all['FoodCourt'] != 0), 'FoodCourt'])

nans = df_all["FoodCourt"].isna()
newvalues = np.random.randint(df_all["FoodCourt"].mean() - df_all["FoodCourt"].std(), df_all["FoodCourt"].mean()
                              + df_all["FoodCourt"].std(), len(nans))
newvalues = np.where(newvalues < 0, 0, newvalues)
df_all["FoodCourt"][np.isnan(df_all["FoodCourt"])] = newvalues

# ShoppingMall


df_all.loc[(df_all['ShoppingMall'] != 0), 'ShoppingMall'] = np.log1p(df_all.loc[(df_all['ShoppingMall'] != 0),
                                                                                'ShoppingMall'])

nans = df_all["ShoppingMall"].isna()
newvalues = np.random.randint(df_all["ShoppingMall"].mean() - df_all["ShoppingMall"].std(), df_all["ShoppingMall"].mean()
                              + df_all["ShoppingMall"].std(), len(nans))
newvalues = np.where(newvalues < 0, 0, newvalues)
df_all["ShoppingMall"][np.isnan(df_all["ShoppingMall"])] = newvalues

# Spa

df_all.loc[(df_all['Spa'] != 0), 'Spa'] = np.log1p(df_all.loc[(df_all['Spa'] != 0), 'Spa'])

nans = df_all["Spa"].isna()
newvalues = np.random.randint(df_all["Spa"].mean() - df_all["Spa"].std(), df_all["Spa"].mean()
                              + df_all["Spa"].std(), len(nans))
newvalues = np.where(newvalues < 0, 0, newvalues)
df_all["Spa"][np.isnan(df_all["Spa"])] = newvalues

# VRDeck

df_all.loc[(df_all['VRDeck'] != 0), 'VRDeck'] = np.log1p(df_all.loc[(df_all['VRDeck'] != 0), 'VRDeck'])

nans = df_all["VRDeck"].isna()
newvalues = np.random.randint(df_all["VRDeck"].mean() - df_all["VRDeck"].std(), df_all["VRDeck"].mean()
                              + df_all["VRDeck"].std(), len(nans))
newvalues = np.where(newvalues < 0, 0, newvalues)
df_all["VRDeck"][np.isnan(df_all["VRDeck"])] = newvalues


# Dropping unnecessary columns

df_all.drop(columns=['CabinN', 'SurnameN'], inplace=True)


print(df_all.info())
print(df_all.corr())
print(df_all.describe())
print(df_all.skew())

df_all.to_csv('Data/df_all_V1', index=True)
