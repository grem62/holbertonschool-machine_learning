#!/usr/bin/env python3

import os
import numpy as np

Deep = __import__('18-deep_neural_network').DeepNeuralNetwork

script_directory = os.path.dirname(os.path.realpath(__file__))
chemin_train = os.path.join(script_directory, 'data', 'Binary_Train.npz')
lib_train = np.load(chemin_train)
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
deep._DeepNeuralNetwork__weights['b1'] = np.ones((5, 1))
deep._DeepNeuralNetwork__weights['b2'] = np.ones((3, 1))
deep._DeepNeuralNetwork__weights['b3'] = np.ones((1, 1))
A, cache = deep.forward_prop(X)
print(A)
print(cache)
print(cache is deep.cache)
print(A is cache['A3'])
