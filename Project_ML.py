import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer,OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers, optimizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping 
from sklearn.compose import ColumnTransformer
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.svm import SVC
import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

# Part 1-data examination 

# Load the dataset
data_train = pd.read_csv('/content/x_train.csv')

# Examine data types
print(data_train.dtypes)

#Check the shape of the training dataset
print(data_train.shape)

# check for missing values 
print(data_train.isna().sum())

# visualize data distribution using histograms or boxplots
data_train.hist()

# get a list of column names and data types in roder to see which ones to drop 
col_types = data_train.dtypes
print (col_types)

# describe the training data 
print(data_train.describe())


#######################################################################################################
# Part 2-Preprocessing steps

#Drop the columns which are not for use

columns_to_drop=['Unnamed: 0','m_power','Tosc', 'Tmix']
data_train = data_train.drop(columns_to_drop, axis=1)

# Scale numerical features
num_cols = data_train.select_dtypes(include=['float64']).columns
scaler = StandardScaler()
data_train[num_cols] = scaler.fit_transform(data_train[num_cols])

#######################################################################################################
# Part 3•-Model building and model training

# Separate the features and labels
X_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values.astype(int)

# Convert target labels to binary encoded format
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

# Get predictions on validation set
X_val_pred = model.predict(X_val)
X_val_pred = np.argmax(X_val_pred, axis=1)
y_val_true = np.argmax(y_val, axis=1)

# Print validation set results
print("True classes:", y_val_true)
print("Predicted classes:", X_val_pred)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

# Evaluate the model on training data
test_loss, test_acc = model.evaluate(X_train, y_train)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


# Plot the training and validation loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#######################################################################################################
# Part 4-Performance tunning
# Build the model with more batch size  more epochs and regulators and dropouts as well as the early stoppings

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True)
# Train the model 

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])


# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)
#Validation loss: 0.10264364629983902
#Validation accuracy: 0.9684244990348816

#######################################################################################################
 # Partie 5-Model evaluation - check how well the model performs on the testing dataset

# Load the dataset
data_test = pd.read_csv('/content/x_test.csv')
print(data_test.columns)

# Drop columns not used in testing

columns_to_drop=['Unnamed: 0','m_power','Tosc', 'Tmix']
data_test = data_test.drop(columns_to_drop, axis=1)

# Drop columns not used in testing
num_cols = data_test.select_dtypes(include=['float64']).columns
scaler = StandardScaler()
data_test[num_cols] = scaler.fit_transform(data_test[num_cols])

# Separate the features and labels
X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].values.astype(int)

#I printed the shape of X_test and y-test which is 3840 rows
#print(X_test.shape)
#print(y_test.shape)

# Convert target labels to binary encoded format
lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)

# Split the dataset into testing and validation sets
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=0)

# Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_test.shape[1]))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(y_test.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_test, y_test, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

# Evaluate the model on testing data
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

##Plot the testing and validation loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('testing and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the testing and validation accuracy

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('testing and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#the result i got
#Validation loss: 0.1556047946214676 ---Validation accuracy: 0.9544270634651184
#Test loss: 0.09221438318490982  -----Test accuracy: 0.9733073115348816
# Get predictions on testing set
X_test_pred = model.predict(data_test.iloc[:, :-1])
X_test_pred = X_test_pred.reshape(-1, y_test.shape[1])
X_test_pred = np.argmax(X_test_pred, axis=1)

# Save predictions to CSV file
result_test = pd.DataFrame({'id': np.arange(len(X_test_pred)), 'target': X_test_pred})
result_test.to_csv('test_result.csv', index=False)



#######################################################################################################
# Part 6-Compare MLP to SVM

#SVM validation accuracy: 0.9680989583333334
#SVM training time: 2.691427707672119
#SVM accuracy: 0.9680989583333334
#MLP training time: 24.474051475524902
#MLP accuracy: 0.96875

##Loading my traning dataset and doing the same processing
data_train = pd.read_csv('/content/x_train.csv')

# Drop the columns which are not for use
columns_to_drop=['Unnamed: 0','m_power','Tosc', 'Tmix']
data_train = data_train.drop(columns_to_drop, axis=1)

# Scale numerical features
num_cols = data_train.select_dtypes(include=['float64']).columns
scaler = StandardScaler()
data_train[num_cols] = scaler.fit_transform(data_train[num_cols])

# Separate the features and labels
X_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values.astype(int)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Build and train the SVM model with different hyperparameters
svm = SVC(kernel='rbf', C=1, gamma='scale')
start_time = time.time()
svm.fit(X_train, y_train)
end_time = time.time()

# Get predictions on validation set
y_val_pred = svm.predict(X_val)

# Calculate accuracy score on validation set
val_acc = accuracy_score(y_val, y_val_pred)

# Calculate training time
training_time = end_time - start_time

# Print SVM results
print("SVM validation accuracy:", val_acc)
print("SVM training time:", training_time)

# Evaluate the SVM model on the validation set
x_pred_svm = svm.predict(X_val)
svm_accuracy = accuracy_score(y_val, x_pred_svm)
print("SVM accuracy:", svm_accuracy)

# Train an MLP model with the same architecture and hyperparameters as in your code
mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', alpha=0.01, batch_size=32, max_iter=500, validation_fraction=0.2)
start_time = time.time()
mlp.fit(X_train, y_train)
print("MLP training time:", time.time()-start_time)

# Evaluate the MLP model on the validation set
x_pred_mlp = mlp.predict(X_val)
mlp_accuracy = accuracy_score(y_val, x_pred_mlp)
print("MLP accuracy:", mlp_accuracy)




#######################################################################################################
# Part 7 -Feature engineering 

##need to install scikeras in google colab in order for the code to work
!pip install scikeras

# Load data
data_train = pd.read_csv('/content/x_train.csv')

# Drop unnecessary columns
columns_to_drop=['Unnamed: 0','m_power','Tosc', 'Tmix']
data_train = data_train.drop(columns_to_drop, axis=1)

# Scale numerical features
num_cols = data_train.select_dtypes(include=['float64']).columns
scaler = StandardScaler()
data_train[num_cols] = scaler.fit_transform(data_train[num_cols])

# Separate features and labels
X_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Feature selection using Random Forest
clf = RandomForestRegressor(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
sfm = SelectFromModel(clf, threshold=0.1)

# Experiment with different number of features
n_features = [2, 3, 5, 7]
val_list = []

for n in n_features:
    sfm = SelectFromModel(clf, threshold='mean', max_features=n)
    sfm.fit(X_train, y_train)
    X_train_new = sfm.transform(X_train)
    X_val_new = sfm.transform(X_val)

    # Build the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train_new.shape[1]))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='linear'))
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    # Train the model
    history = model.fit(X_train_new, y_train, epochs=50, batch_size=32, validation_data=(X_val_new, y_val), verbose=0)
    
    # Evaluate the model on validation data
    val_ev_mse, val_metric = model.evaluate(X_val_new, y_val, verbose=0)
    val_list.append(val_ev_mse)
    print("Number of features:", n, "Validation MSE:", val_ev_mse)

plt.plot(n_features, val_list)
plt.title('Validation MSE vs. Number of features')
plt.ylabel('Validation MSE')
plt.xlabel('Number of features')
plt.show()

#Number of features: 2 Validation MSE: 0.0796009823679924
#Number of features: 3 Validation MSE: 0.06318943947553635
#Number of features: 5 Validation MSE: 0.060656312853097916
#Number of features: 7 Validation MSE: 0.062497127801179886


#These results mean that as the number of selected features increases, the validation MSE decreases
#. This suggests that using more features leads to a more accurate regression model. 
#However, it is important to note that this trend may not always hold,
#as selecting too many features can lead to overfitting and a decrease in model performance on new data. 
#Therefore, it is important to strike a balance between the number of features used 
#and the performance of the model. In this case, it seems that using 5 features gives 
#the best trade-off between model complexity and performance.