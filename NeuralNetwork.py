import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
# data['CLASS'].unique()
# data['Gender'].unique()

# split and target variable
Y = data['CLASS'].values
X = data.drop(columns=['CLASS']).values
# data.to_csv('Diabetes4hope.csv', index=False)

encoder = OneHotEncoder(categories='auto', sparse_output=False)
Y = encoder.fit_transform(Y.reshape(-1, 1))  # Reshape Y to make it 2D
Y = Y[:, 1:]
print(Y)
# X = np.delete(X, 1, axis=1)

# split to training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=12)


X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(units=128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.5),
    Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dense(units=1, activation='sigmoid')
])

# classifier = Sequential()
# classifier.add(Dense(units=64, activation='relu', input_dim=X.shape[1]))
# classifier.add(Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# classifier.add(BatchNormalization())
# classifier.add(Dense(units=1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.1)

Y_pred = model.predict(X_test)
Y_pred_int = (Y_pred > 0.5).astype(int)

cm = confusion_matrix(Y_test, Y_pred_int)
acc = accuracy_score(Y_test, Y_pred_int)

history = model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, Y_test))

# training and accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("Accuracy:", acc)
print("Alice Matrix:\n", cm)
