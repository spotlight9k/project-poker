import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
import time
import pandas as pd

# Load training data
x_train_list = []
with open('poker_train.txt', 'r') as inputfile:
    for row in csv.reader(inputfile):
        x_train_list.append(row)

# Convert x_train from list to np_array
x_train = np.asarray(x_train_list)

# Extract class labels
y_train = x_train[:, [10]]
y_train = y_train.reshape(25010)
y_train = y_train.astype(np.int32)

# Delete class labels from x_train
x_train = x_train[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

# Creating np.arrays for sort
new_suit = np.zeros(4)
new_rank = np.zeros(13)
new_hand = np.zeros((85,))
new_x_train = np.zeros((25010, 85))
new_y_train = np.zeros((25010, 10))
new_x_test = np.zeros((1000000, 85))
new_y_test = np.zeros((1000000, 10))

print(new_suit.shape, new_suit)
print(new_rank.shape, new_rank)
print(new_hand.shape)

x_train = x_train.astype(np.int32)

for i in range(0, 25010):
    suit_counter = 0
    for j in range(0, 10):
        if j % 2 == 0:
            new_suit = np.zeros(4)
            new_suit[x_train[i, j] - 1] = 1
            new_hand[suit_counter:suit_counter] = new_suit
            rank_counter = rank_counter + 4
        else:
            new_rank = np.zeros(13)
            new_rank[x_train[i, j] - 1] = 1
            new_hand[suit_counter:suit_counter + 13] = new_rank
            suit_counter = suit_counter + 13
    new_x_train[i, :] = new_hand

print("new_x_train shape: ", new_x_train.shape)

for i in range(0, 25010):
    new_class = np.zeros(10)
    new_class[y_train[i] - 1] = 1
    new_y_train[i, :] = new_class

print("new_y_train shape: ", new_y_train.shape)

# Convert x_train from float64 to int32
x_train = x_train.astype(np.int32)

# MLP Classification
print("Fitting data...")

HIDDEN_LAYERS_NUMBER = 110
ACT_FUNCTION = 'tanh'
CONST_SOLVER = 'adam'
CONST_ALPH = 0.0001
CONST_ITER = 70

clf = MLPClassifier(activation=ACT_FUNCTION,
                    solver=CONST_SOLVER,
                    alpha=CONST_ALPH,
                    hidden_layer_sizes=(HIDDEN_LAYERS_NUMBER, HIDDEN_LAYERS_NUMBER, HIDDEN_LAYERS_NUMBER),
                    max_iter=CONST_ITER,
                    verbose=True,
                    warm_start=True,
                    learning_rate='adaptive',
                    tol=1e-8)

print(clf.get_params(deep=True))
clf.fit(new_x_train, new_y_train)
print(clf.fit(new_x_train, new_y_train))

accuracy_train = clf.score(new_x_train, new_y_train)
loss_train = clf.loss_

# Load testing data
x_test_list = []
with open('poker_test.txt', 'r') as inputfile:
    for row in csv.reader(inputfile):
        x_test_list.append(row)

# Convert x_test from list to np_array
x_test = np.asarray(x_test_list)

# Extract class labels
y_test = x_test[:, [10]]
y_test = y_test.reshape(1000000)
y_test = y_test.astype(np.int32)

for i in range(0, 1000000):
    new_class = np.zeros(10)
    new_class[y_test[i] - 1] = 1
    new_y_test[i, :] = new_class

print("new_y_test shape: ", new_y_test.shape)

# Delete class labels from x_test
x_test = x_test[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
x_test = x_test.astype(np.int32)

for i in range(0, 1000000):
    rank_counter = 0
    for j in range(0, 10):
        if j % 2 == 0:
            new_suit = np.zeros(4)
            new_suit[x_test[i, j] - 1] = 1
            new_hand[rank_counter:rank_counter + 4] = new_suit
            rank_counter = rank_counter + 4
        else:
            new_rank = np.zeros(13)
            new_rank[x_test[i, j] - 1] = 1
            new_hand[rank_counter:rank_counter + 13] = new_rank
            rank_counter = rank_counter + 13
    new_x_test[i, :] = new_hand

print("new_X_test shape: ", new_x_test.shape)

# Score x_test
start = time.time()

accuracy_test = clf.score(new_x_test, new_y_test) * 100
loss_test = clf.loss_

end = time.time()
WT = end - start

new_row = {'H.L.N.': HIDDEN_LAYERS_NUMBER * 3,
           'activation': ACT_FUNCTION,
           'solver': CONST_SOLVER, 'alpha': CONST_ALPH,
           'loss_tr': round(loss_train, 3),
           'accuracy_tr': round(accuracy_train, 3),
           'loss_ts': round(loss_test, 3),
           'accuracy_ts': round(accuracy_test, 3),
           'work time': round(WT, 3), 'iter': iter}

data = pd.read_csv('tstr1_data.txt', delimiter=',')
data = data.append(new_row, ignore_index=True)
data.to_csv('tstr1_data.txt', index=False)