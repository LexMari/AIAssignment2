import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('C:/Users/legos/Downloads/Dataset of Diabetes.csv')

dataset['CLASS'] = dataset['CLASS'].replace({'N ': 'N', 'Y ': 'Y'})
dataset['Gender'] = dataset['Gender'].replace({'f': 'F'})

# split dataset into X and Y
X = pd.DataFrame(dataset.iloc[:, 2:13].values)
Y = dataset.iloc[:, 13].values

# encode categorical data
label_encoder_X_0 = LabelEncoder()
X.loc[:, 0] = label_encoder_X_0.fit_transform(X.iloc[:, 0])

# one hot encoding
encoder = OneHotEncoder(categories='auto', sparse_output=False)
Y = encoder.fit_transform(Y.reshape(-1, 1))  # Reshape Y to make it 2D
print(Y)

# split X Y dataset into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# perform feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model architecture for binary classification
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
# Use 'sigmoid' for binary classification
classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
classifier.fit(X_train, Y_train, batch_size=10, epochs=80)

# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# cm = confusion_matrix(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)

# print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)

