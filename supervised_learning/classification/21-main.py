#!/usr/bin/env python3

import numpy as np
import os

Deep = __import__('21-deep_neural_network').DeepNeuralNetwork

script_directory = os.path.dirname(os.path.realpath(__file__))
chemin_train = os.path.join(script_directory, 'data', 'Binary_Train.npz')
lib_train = np.load(chemin_train)
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
A, cache = deep.forward_prop(X)
deep.gradient_descent(Y, cache, 0.5)
print(deep.weights)
