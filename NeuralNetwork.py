import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/legos/Downloads/Dataset of Diabetes.csv')
data.info()

# drop irrelevant columns
dropcols = ['ID', 'No_Pation']
data = data.drop(dropcols, axis=1)
data.info()

X = data.values
Y = data['CLASS'].values

# create dummy variables
dummies = []
dummycols = ['Gender', 'CLASS']
for dummycol in dummycols:
    dummies.append(pd.get_dummies(data[dummycol]))

diabetes_dummies = pd.concat(dummies, axis=1)

# need to group F and f together
data = pd.concat([data, diabetes_dummies], axis=1)
data = data.drop(['Gender', 'CLASS'], axis=1)
data.info()

X = np.delete(X, 1, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)