import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

def fillmissingvalues(df, columns, variable, wages):
    values = []
    df_nulls = df.loc[df[[variable]].isnull().all(axis=1)]
    df_counts = pd.DataFrame(index=df[variable].unique())
    for nr in range(len(df_nulls)):
        n = 0
        for column in columns:
            observation = df_nulls[column].iloc[nr]
            df_count = df.loc[df[column] == observation].groupby([variable])[variable].count().sort_values(
                ascending=False)
            df_count = pd.DataFrame({nr: df_count}, index=df[variable].unique())
            df_count[nr] = df_count[nr].apply(lambda x: x/df_count[nr].sum()*wages[n])
            df_counts = pd.concat([df_counts, df_count], axis=1)
            n = n + 1
        df_counts['sum'] = df_counts.sum(axis=1)
        print(df_counts['sum'].sort_values(ascending=False))
        values.append(df_counts['sum'].sort_values(ascending=False).index.values[0])
        df_counts = df_counts.drop(df_counts.columns, 1)
    df.loc[df[[variable]].isnull().all(axis=1), variable] = values
    return df

def getlinearregressionmodel(dataframe, Xx, yy):
    X = dataframe[Xx]
    y = dataframe[yy]
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

def dropcolumnstesttrain(df_train, df_test, columns):
    return df_train.drop(columns=columns), df_test.drop(columns=columns)

def extratreefeatureselection(Xy, yname, times):
    w = pd.DataFrame(Xy.drop(columns=[yname]).columns)
    model = ExtraTreesClassifier()
    Xy = Xy[~Xy.isnull().any(axis=1)]
    print(len(Xy))
    X = Xy.drop(columns=[yname]).values
    X = preprocessing.scale(X)
    y = Xy[yname].values
    for n in range(times):
        model.fit(X, y)
        w[str(n+1)] = model.feature_importances_

    w['mean'] = w.select_dtypes(include=np.number).mean(axis=1)
    w.rename(columns={0: 'variable'}, inplace=True)
    print(w[['variable', 'mean']])
    return w[['variable', 'mean']]

def getnoimportantvar(data, value):
    return data.loc[data["mean"] < value, 'variable'].tolist()

def testmodel(df_train, model, y, percent):

    X = df_train.drop(columns=[y]).values
    X = preprocessing.scale(X)
    y = df_train[y].values

    length = int(percent*len(df_train))

    X_train = X[:-length]
    y_train = y[:-length]

    X_test = X[-length:]
    y_test = y[-length:]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Score:', model.score(X_test, y_test))
    error = np.power((y_pred-y_test), 2).sum()/len(y_test)
    # print(error)
    # plt.scatter(y_pred, y_test-y_pred, color="green")
    # plt.show()

def finalscoretranslation(x):
    if x == 1:
        return True
    else:
        return False


def finalprediction(model, df_train, df_test, y, df_submission):

    X = df_train.drop(columns=[y]).values
    X = preprocessing.scale(X)
    yy = df_train[y].values

    model.fit(X, yy)

    X_test = preprocessing.scale(df_test)
    y_pred = model.predict(X_test)
    df_submission[y] = y_pred
    df_submission[y] = df_submission[y].apply(lambda x: finalscoretranslation(x))
    df_submission.to_csv('Data/submission.csv', index=True)