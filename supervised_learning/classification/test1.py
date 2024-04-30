#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

Neuron = __import__('7-neuron').Neuron

script_directory = os.path.dirname(os.path.realpath(__file__))
chemin_train = os.path.join(script_directory, 'data', 'Binary_Train.npz')
lib_train = np.load(chemin_train)
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

np.random.seed(1)
neuron = Neuron(X_train.shape[0])
neuron.train(X_train, Y_train, iterations=1000)
