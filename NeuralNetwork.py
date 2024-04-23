import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('C:/Users/legos/Downloads/Dataset of Diabetes.csv')

# drop irrelevant columns
dropcols = ['ID', 'No_Pation']
data = data.drop(dropcols, axis=1)

# Filter rows to only contain 'N', 'Y', and 'P' in 'CLASS' column
data['CLASS'] = data['CLASS'].replace({'N ': 'N', 'Y ': 'Y'})
data['Gender'] = data['Gender'].replace({'f': 'F'})

# Encode 'CLASS' and 'Gender' columns
label_encoder = preprocessing.LabelEncoder()
data['CLASS'] = label_encoder.fit_transform(data['CLASS'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['CLASS'].unique()
data['Gender'].unique()

Y = data['CLASS'].values
X = data.drop(columns=['CLASS']).values
# data.to_csv('Diabetes4hope.csv', index=False)

X = np.delete(X, 1, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

classifier = Sequential()
classifier.add(Dense(units=64, activation='relu', input_dim=X.shape[1]))
classifier.add(Dense(units=32, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(units=1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1)

Y_pred = classifier.predict(X_test)
Y_pred_int = (Y_pred > 0.5).astype(int)

cm = confusion_matrix(Y_test, Y_pred_int)
acc = accuracy_score(Y_test, Y_pred_int)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
