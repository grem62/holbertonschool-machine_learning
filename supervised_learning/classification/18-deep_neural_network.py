#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


class DeepNeuralNetwork:
    """Class that defines a deep neural network performing
    """

    def __init__(self, nx, layers):
        """Initialization of a deep neural network

        Args:
            nx (int): number of input features to the neuron
            layers (list(int)): list representing the number of nodes in each
            layer of the network
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')

        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        if min(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if i == 0:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
                self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """_summary_"""
        return self.__L

    @property
    def cache(self):
        """_summary_"""
        return self.__cache

    @property
    def weights(self):
        """_summary_"""
        return self.__weights

    def forward_prop(self, X):
        """_summary_

        Args:
            X (_type_): _description_
        """
        self.__cache['A0'] = X
        for i in range(self.L):
            z = np.matmul(self.weights['W' + str(i + 1)],
                          self.cache['A' + str(i)]) + self.weights['b' +
                                                                   str(i + 1)]
            self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-z))
        return self.cache['A' + str(self.L)], self.cache
