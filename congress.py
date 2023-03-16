import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt

def train_test_split(pd_data: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    pd_dataset = pd_data.copy()
    pd_dataset = pd_dataset[pd_dataset.columns[1:]]
    index = np.arange(len(pd_dataset))
    index = np.random.permutation(index)
    train_ammount = int(len(index)*test_ratio)
    train_ids = index[train_ammount:]
    test_ids = index[:train_ammount]
    
    train_dataset = pd_dataset[pd_dataset.index.isin(train_ids)].reset_index()
    test_dataset = pd_dataset[pd_dataset.index.isin(test_ids)].reset_index()
    
    train_dataset = train_dataset[train_dataset.columns[1:]]
    test_dataset = test_dataset[test_dataset.columns[1:]]

    return train_dataset[train_dataset.columns[1:]], train_dataset[train_dataset.columns[0]], test_dataset[test_dataset.columns[1:]], test_dataset[test_dataset.columns[0]]

path_to_dataset = 'LAB5-NeuralNetwork/data.csv' # change the PATH
pd_dataset = pd.read_csv(path_to_dataset)
pd_dataset = pd_dataset.replace("?", np.nan)
pd_dataset = pd_dataset.fillna(pd_dataset.mode().iloc[0])

pd_dataset.replace("republican", 0.0, inplace=True)
pd_dataset.replace("democrat", 1.0, inplace=True)

pd_dataset.replace("y", True, inplace=True)
pd_dataset.replace("n", False, inplace=True)


x_train, y_train, x_test, y_test = train_test_split(pd_dataset)

model = Sequential()
model.add(Dense(12, input_shape=(16,), activation='sigmoid'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=200, batch_size=15)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy*100))

plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('n epochs')
plt.ylabel('loss')
plt.show()
