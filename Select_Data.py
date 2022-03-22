import pandas as pd
from Functions import *


pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 200)

df_all = pd.read_csv('Data/df_all_V3', index_col=0)

df_train = df_all.loc[df_all[['Transported']].notnull().any(axis=1)]
df_test = df_all.loc[df_all[['Transported']].isnull().any(axis=1)]
df_test = df_test.drop(columns=['Transported'])

mapping = {False: 0, True: 1}
df_train['Transported'] = df_train['Transported'].map(mapping).astype(int)

print(df_train.info())
print(df_test.info())
print(df_train.corr())

# Using Extra Trees Classifier feature importance for dropping some less useful variables

df_var_import = extratreefeatureselection(df_train.select_dtypes(include=np.number), 'Transported', 3)
df_train, df_test = dropcolumnstesttrain(df_train, df_test, getnoimportantvar(df_var_import, 0.025))

df_var_import = extratreefeatureselection(df_train.select_dtypes(include=np.number), 'Transported', 1)

df_train.to_csv('Data/df_train', index=True)
df_test.to_csv('Data/df_test', index=True)