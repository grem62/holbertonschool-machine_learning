#!/usr/bin/env python3
"""_summary_
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def cost(self, Y, A):
        """_summary_

        Args:
            Y (_type_): _description_
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        m = Y.shape[1]
        return np.sum(-(Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        A, i = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """_summary_

        Args:
            Y (_type_): _description_
            cache (_type_): _description_
            alpha (float, optional): _description_. Defaults to 0.05.
        """
        m = Y.shape[1]
        dz = cache['A' + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            dw = (1 / m) * np.matmul(dz, cache['A' + str(i - 1)].T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dz = np.matmul(self.weights['W' + str(i)].T, dz) * (
                cache['A' + str(i - 1)] * (1 - cache['A' + str(i - 1)]))
            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            iterations (int, optional): _description_. Defaults to 5000.
            alpha (float, optional): _description_. Defaults to 0.05.
            verbose (bool, optional): _description_. Defaults to True.
            graph (bool, optional): _description_. Defaults to True.
            step (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')

        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        if graph or verbose:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        cost_list = []
        itterations_list = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    print('Cost after {} iterations: {}'.format(i, cost))
                if graph:
                    cost_list.append(i)
                    itterations_list.append(cost)
        if graph:
            plt.plot(cost_list, itterations_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """_summary_

        Args:
            filename (_type_): _description_
        """
        if type(filename) is not str:
            return None
        if filename != '.pkl':
            filename = filename + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """_summary_

        Args:
            filename (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
