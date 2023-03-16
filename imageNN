import os
os.environ["tf_gpu_allocator"]="cuda_malloc_async"
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.regularizers import L2
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import datetime

tf.config.list_physical_devices("GPU")

def model_builder(hp):
    model = Sequential()

    """ model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu')) """

    filters_layer_1 = hp.Int("layers_1", min_value=16, max_value=64, step=16)
    filters_layer_2 = hp.Int("layers_2", min_value=32, max_value=128, step=16)
    filters_layer_3 = hp.Int("layers_3", min_value=32, max_value=128, step=16)



    model.add(Conv2D(filters= filters_layer_1, kernel_size=(3,3), activation ='relu', input_shape= (32,32,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(filters_layer_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters_layer_3, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))

    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])

    return model

font = {'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
labels = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

X_train_scaled = X_train.astype('float32') / 255.0
y_train_encoded = to_categorical(y_train, num_classes=10)

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='Tuner',
                     project_name='Lab6')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

stop_early = EarlyStopping(monitor='val_loss', patience=10)
tuner.search(X_train_scaled, y_train_encoded, epochs=50, validation_split=0.2, callbacks=[stop_early, tensorboard_callback], verbose=2)

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_scaled, y_train_encoded, epochs=50, validation_split=0.2, 
    callbacks=[tensorboard_callback], verbose=2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(X_train_scaled, y_train_encoded, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(X_train_scaled, y_train_encoded)
print("[test loss, test accuracy]:", eval_result)
