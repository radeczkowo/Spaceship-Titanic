import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')

print(df_train.info())
print(df_test.info())
print(len(df_train))
print(len(df_test))
print(df_train.tail(30))
print(df_test.head(30))
print(df_train.describe())


# "Transported" variable

df_Tr_percent = df_train['Transported'].value_counts().apply(lambda x: x / len(df_train))
print(df_Tr_percent)
df_Tr_percent.plot(kind='bar')
plt.show()

# The amount of True and False in variable is nearly even

df_all = pd.concat([df_train, df_test])

# PassengerId contain a group data, which can be useful for classification

df_all[["Group", 'Rest']] = df_all["PassengerId"].str.split('_', expand=True)
df_all.drop(columns=['Rest'], inplace=True)

# Persons with the same surnames can come from the same family

df_all[["FirstName", 'Surname']] = df_all["Name"].str.split(' ', expand=True)

df_all.drop(columns=['Name', 'FirstName'], inplace=True)

df_all.to_csv('Data/df_all_V0', index=True)


