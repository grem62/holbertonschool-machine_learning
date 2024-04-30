#!/usr/bin/env python3

import numpy as np
import os

Neuron = __import__('2-neuron').Neuron
script_directory = os.path.dirname(os.path.realpath(__file__))
chemin_train = os.path.join(script_directory, 'data', 'Binary_Train.npz')
lib_train = np.load(chemin_train)
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
neuron._Neuron__b = 1
A = neuron.forward_prop(X)
if (A is neuron.A):
        print(A)
