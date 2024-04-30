#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

NN = __import__('15-neural_network').NeuralNetwork

script_directory = os.path.dirname(os.path.realpath(__file__))
chemin_train = os.path.join(script_directory, 'data', 'Binary_Train.npz')
lib_train = np.load(chemin_train)
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
script_directory = os.path.dirname(os.path.realpath(__file__))
chemin_train = os.path.join(script_directory, 'data', 'Binary_Dev.npz')
lib_dev = np.load(chemin_train)
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
