import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
# Load the dataset
df = pd.read_csv('/content/x_train.csv')

# Check for missing values
print(df.isnull().sum())

# Replace missing values in numerical columns with median
num_cols = ['Unnamed: 0', 'cfo_demod', 'gain_imb', 'iq_imb', 
            'or_off', 'quadr_err', 'm_power', 'ph_err', 'mag_err', 'evm', 'Tosc', 'Tmix']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Scale the numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encode the categorical features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_cols)], remainder='passthrough')
df = ct.fit_transform(df)

# Split the data into train and test sets
X = df[:, :-1]
y = df[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train to integer type
y_train = y_train.astype(int)

# Balance the data using SMOTE (oversampling)
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)



#######################################################################################################
# Partie 3

## I got these results : Validation loss: 0.02260235883295536 -- Validation accuracy: 0.990234375

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, to_categorical(y_train),
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test))
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Plot the train and validation loss changes during the training epochs
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss_train)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#######################################################################################################
# Partie 4

## I got these results Test accuracy: 0.9860026240348816 --Test loss: 0.04700838029384613
# Build the model with regularization techniques
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Add early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(X_train, to_categorical(y_train),
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test))
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Plot the train and validation loss changes during the training epochs
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss_train)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#######################################################################################################
# Partie 5 Model evaluation - check how well the model performs on the testing dataset

#I got these results in this part (Test accuracy: 0.9820312261581421 -Test loss: 0.05022028461098671)

# Load the test dataset 
test_df = pd.read_csv('/content/x_test.csv')

# Replace missing values in numerical columns with median
for col in num_cols:
    test_df[col].fillna(test_df[col].median(), inplace=True)

# Scale the numerical features
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Encode the categorical features
test_df = ct.transform(test_df)

# Split the data into X and y
X_test = test_df[:, :-1]
y_test = test_df[:, -1]

# Convert y_test to integer type
y_test = y_test.astype(int)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test))
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

#######################################################################################################
# Part 6

#I got these results for this part
#Training time for SVM: 2.16 seconds
#SVM test accuracy: 0.9697916666666667
#Training time for MLP: 143.92 seconds
#MLP test accuracy: 0.9869791865348816   
#MLP test loss: 0.03310440480709076

# Train an SVM
start_time = time.time()
svm = SVC(kernel='rbf', C=1.0, gamma='auto')
svm.fit(X_train, y_train)
end_time = time.time()
print(f"Training time for SVM: {end_time - start_time:.2f} seconds")

# Evaluate the SVM on the test set
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print('SVM test accuracy:', acc_svm)

# Train the MLP
from keras.models import Sequential
from keras.layers import Dense

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model.fit(X_train, to_categorical(y_train),
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)
end_time = time.time()
print(f"Training time for MLP: {end_time - start_time:.2f} seconds")

# Evaluate the MLP on the test set
test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test))
print('MLP test accuracy:', test_acc)
print('MLP test loss:', test_loss)

#######################################################################################################
# Part 7 #Feature engineering -
#  i had this as a result : Test accuracy: 0.693359375 -Test test_loss: 0.8650355935096741

# Load the dataset
df = pd.read_csv('/content/x_train.csv')

# Check for missing values
print(df.isnull().sum())

# Replace missing values in numerical columns with median
num_cols = ['Unnamed: 0', 'cfo_demod', 'gain_imb', 'iq_imb', 
            'or_off', 'quadr_err', 'm_power', 'ph_err', 'mag_err', 'evm', 'Tosc', 'Tmix']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Drop some of the categorical and numerical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = cat_cols[2:] # drop the first two categorical columns
num_cols = num_cols[1:] # drop one numerical column

# Scale the numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Encode the categorical features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_cols)], remainder='passthrough')
df = ct.fit_transform(df)

# Split the data into train and test sets
X = df[:, :-1]
y = df[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train to integer type
y_train = y_train.astype(int)

# Balance the data using SMOTE (oversampling)
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, to_categorical(y_train),
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test))
print('Test accuracy:', test_acc)
print('Test test_loss:', test_loss)