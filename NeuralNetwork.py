import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('C:/Users/legos/Downloads/Dataset of Diabetes.csv')

X = pd.DataFrame(dataset.iloc[:, 2:13].values)
Y = dataset.iloc[:, 13].values

labelencoder_X_0 = LabelEncoder()
X.iloc[:, 0] = labelencoder_X_0.fit_transform(X.iloc[:, 0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize the classifier
classifier = Sequential()

# Add the first hidden layer with 6 neurons, ReLU activation, and input dimension of 13
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=13))

# Add the second hidden layer with 6 neurons and ReLU activation
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Add the output layer with 1 neuron and sigmoid activation for binary classification
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the model with Adam optimizer and binary cross-entropy loss
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
classifier.fit(X_train, Y_train, epochs=150, batch_size=10)

# Display model summary
classifier.summary()

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(Y_test, y_pred)
print(cm)
accuracy_score(Y_test, y_pred)
