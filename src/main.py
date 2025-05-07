#import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

#load data
train_data = pd.read_csv(r'C:\Users\saiga\OneDrive\Desktop\Projects\Train.csv')
print("Shape of train_data:", train_data.shape)


X = train_data.iloc[:, 1:]  
y = train_data.iloc[:, 0]   

print("Shape of X after separating features:", X.shape)

#Preprocess the Data
if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  
X = X.values / 255.0
X = X.reshape(-1, 28, 28, 1)
print("Shape of X after reshaping:", X.shape)

#One-Hot Encode the Labels
y = to_categorical(y, num_classes=10)
print("Shape of y after one-hot encoding:", y.shape)

#Split the Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
print("X_train shape:", X_train.shape)

#Build the Neural Network Model
model = Sequential([
    Flatten(input_shape=(28, 28, 1)), 
    Dense(128, activation='relu'),     
    Dense(64, activation='relu'),      
    Dense(10, activation='softmax')    
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

#Evaluate the Model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

#Make Predictions
test_data = pd.read_csv(r'C:\Users\saiga\OneDrive\Desktop\Projects\Train.csv')
X_test = test_data.drop('label', axis=1).values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
for i in range(5):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis('off')
    plt.show()
