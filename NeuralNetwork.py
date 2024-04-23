import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('C:/Users/legos/Downloads/Dataset of Diabetes.csv')

# drop irrelevant columns
dropcols = ['ID', 'No_Pation']
data = data.drop(dropcols, axis=1)

# Filter rows to only contain 'N', 'Y', and 'P' in 'CLASS' column
filterclassdata = data[data['CLASS'].isin(['N', 'Y', 'P'])]

# Filter 'Gender' column to only contain 'M' and 'F'
filtered_dataset = filterclassdata[filterclassdata['Gender'].isin(['M', 'F'])]

# Reset index
filtered_dataset.reset_index(drop=True, inplace=True)

# Encode 'CLASS' and 'Gender' columns
label_encoder = preprocessing.LabelEncoder()
filtered_dataset['CLASS'] = label_encoder.fit_transform(filtered_dataset['CLASS'])
filtered_dataset['Gender'] = label_encoder.fit_transform(filtered_dataset['Gender'])
filtered_dataset['CLASS'].unique()
filtered_dataset['Gender'].unique()


X = filtered_dataset.values
Y = filtered_dataset['CLASS'].values
filtered_dataset.to_csv('Diabetes3.csv', index=False)

X = np.delete(X, 1, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

tf.convert_to_tensor(X_train, dtype=tf.float32)
tf.convert_to_tensor(Y_train, dtype=tf.float32)

classifier = Sequential()
classifier.add(Dense(units=10, activation='relu', input_dim=X.shape[1]))
classifier.add(Dense(units=10, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, epochs=100, batch_size=10)

Y_pred = classifier.predict(X_test)
Y_pred_int = (Y_pred > 0.5).astype(int)
cm = confusion_matrix(Y_test, Y_pred_int)
acc = accuracy_score(Y_test, Y_pred_int)
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
