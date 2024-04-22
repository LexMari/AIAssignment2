
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

data = pd.read_csv('C:/Users/legos/Downloads/Dataset of Diabetes.csv',
                  dtype=np.float32, delimiter=',')

X = data.drop(columns=['ID', 'No_Pation'])
Y = data['CLASS']  # Keep the CLASS column

print(data, data.shape)  # (759, 9)

x = data[:, 2:13]
print(x[:2])
print(Y[:13])

model = Sequential([
    Dense(units=64, input_dim=8, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid'),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.fit(x, y, epochs=200, batch_size=32, verbose=2)
print(model.evaluate(x, y))

print()


def build_model():
    model = Sequential()
    model.add(Dense(units=64, input_dim=8, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


estimatorModel = Sequential(build_fn=build_model, epochs=200, batch_size=32, verbose=2)
kfold = KFold(n_splits=5, shuffle=True, random_state=12)
print(cross_val_score(estimatorModel, x, y, cv=kfold))
estimatorModel.fit(x, y, epochs=200, batch_size=32, verbose=2)
# print(estimatorModel.evaluate(x, y))

pred = estimatorModel.predict(x[:3, :])
print('예측값 : ', pred.flatten())
print('실제값 : ', y[:3])

print()
from sklearn.metrics import accuracy_score

print('estimatorModel', accuracy_score(y, estimatorModel.predict(x)))
