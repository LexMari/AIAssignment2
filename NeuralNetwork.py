import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('C:/Users/legos/Downloads/Dataset of Diabetes.csv')

# Drop unnecessary columns
X = dataset.drop(columns=['ID', 'No_Pation', 'CLASS'])
Y = dataset['CLASS']

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=['Gender'])

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)

label_encoder = LabelEncoder()
Y_test_encoded = label_encoder.fit_transform(Y_test)

# Model Building
classifier = Sequential()
classifier.add(Dense(units=6, activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dense(units=6, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train_encoded, epochs=999, batch_size=10)

# Model Evaluation
Y_pred = classifier.predict(X_test)
Y_pred_binary = (Y_pred > 0.5)  # Apply threshold for binary prediction
cm = confusion_matrix(Y_test_encoded, Y_pred_binary)
acc = accuracy_score(Y_test_encoded, Y_pred_binary)
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)