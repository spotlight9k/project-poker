import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# random seeds must be set before importing keras & tensorflow
my_seed = 512
np.random.seed(my_seed)

import random

random.seed(my_seed)

import tensorflow as tf

tf.random.set_seed(my_seed)

data_train = pd.read_csv("poker_train_bin", header=None)
data_test = pd.read_csv("poker_test_bin", header=None)
col = ['Card #1', 'Card #2', 'Card #3', 'Card #4', 'Card #5', 'Combination']

data_train.columns = col
data_test.columns = col

y_train = data_train['Combination']
y_test = data_test['Combination']

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

x_train = data_train.drop('Combination', axis=1)
x_test = data_test.drop('Combination', axis=1)

print('Shape of Training Set:', x_train.shape)
print('Shape of Testing Set:', x_test.shape)

from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers

DENSE_1 = 1000
DENSE_2 = 10
ACTIVATION_1 = 'sigmoid'
CONST_OPTIMIZER = 'adamax'

model = Sequential()
model.add(Dense(DENSE_1, activation=ACTIVATION_1, input_dim=5))
model.add(Dense(DENSE_2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=CONST_OPTIMIZER,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=1000,
                    validation_data=(x_test, y_test),
                    verbose=0, shuffle=True)

score = model.evaluate(x_test, y_test, batch_size=10)
loss = round(score[0], 4)
accuracy = round(score[1], 4)

new_row = {'Dense_1': DENSE_1, 'Dense_2': DENSE_2,
           'Activation function_1': ACTIVATION_1,
           'Optimizer': CONST_OPTIMIZER,
           'Loss': loss,
           'Accuracy': accuracy}

accuracy_data = pd.read_csv('accuracy_keras.txt', delimiter=',')
accuracy_data.append({'Dense_1': DENSE_1, 'Dense_2': DENSE_2,
                      'Activation function_1': ACTIVATION_1,
                      'Optimizer': CONST_OPTIMIZER,
                      'Loss': loss,
                      'Accuracy': accuracy}, ignore_index=True)

accuracy_data = accuracy_data.append(new_row, ignore_index=True)
accuracy_data.tail()
accuracy_data.to_csv('accuracy_keras.txt', index=False)
