import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 200)

df_all = pd.read_csv('Data/df_all_V1', index_col=0)

print(df_all.info())
print(df_all.corr())
print(df_all.describe())

# Age Bind

df_all["AgeBinds"] = pd.cut(df_all["Age"], [-1, 10, 20, 30, 40, 50, 60, 1000], labels=["0-10", "10-20", "20-30",
                                                                                        "30-40", "40-50", "50-60",
                                                                                        "70+"])
print(df_all["AgeBinds"].value_counts())

# Spent

df_all['Spent'] = df_all['RoomService'] + df_all['FoodCourt'] + df_all['ShoppingMall'] + df_all['Spa'] + df_all['VRDeck']

print(df_all.skew())

# SurnameGroupSize

group = df_all.groupby(['Group', 'Surname'])['Group'].count()
df_group = pd.DataFrame(group)
df_group.rename(columns={'Group': 'Groups'}, inplace=True)
df_group.reset_index(inplace=True)
df_group.rename(columns={'Groups': 'SurnameGroupSize'}, inplace=True)
print(df_group)
df_all = df_all.merge(df_group, on=['Group', 'Surname'])

# GroupSize

group = df_all.groupby(['Group'])['PassengerId'].count()
df_group = pd.DataFrame(group)
df_group.reset_index(inplace=True)
df_group.rename(columns={'PassengerId': 'GroupCount'}, inplace=True)
df_all = df_all.merge(df_group, on='Group')

print(df_all.info())

df_all.to_csv('Data/df_all_V2', index=True)
