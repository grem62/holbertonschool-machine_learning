#!/usr/bin/env python3
"""_summary_
"""
import numpy as np
import matplotlib.pyplot as plt
"""_summary_

    Raises:
        TypeError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
"""
class DeepNeuralNetwork:
    """Class that defines a deep neural network performing
    """

    def __init__(self, nx, layers):
        """_summary_

        Args:
            nx (_type_): _description_
            layers (_type_): _description_
        """
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W" + str(i + 1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2 / nx)
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        """_summary_
        """
        return self.__L

    @property
    def cache(self):
        """_summary_
        """
        return self.__cache

    @property
    def weights(self):
        """_summary_
        """
        return self.__weights

    def forward_prop(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            self.__cache["A" + str(i)] = self.sigmoid(
                np.matmul(self.__weights["W" + str(i)],
                          self.__cache["A" + str(i - 1)]) +
                self.__weights["b" + str(i)])
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """_summary_

        Args:
            Y (_type_): _description_
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        A, _ = self.forward_prop(X)
        return np.where(A == np.amax(A, axis=0), 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """_summary_

        Args:
            Y (_type_): _description_
            cache (_type_): _description_
            alpha (_type_, optional): _description_. Defaults to 0.05.
        """
        m = Y.shape[1]
        for i in range(self.__L, 0, -1):
            A = cache["A" + str(i)]
            A_prev = cache["A" + str(i - 1)]
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            if i == self.__L:
                dZ = A - Y
            else:
                dZ = np.matmul(W.T, dZ) * (A * (1 - A))
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W" + str(i)] = W - alpha * dW
            self.__weights["b" + str(i)] = b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            iterations (_type_, optional): _description_. Defaults to 5000.
            alpha (_type_, optional): _description_. Defaults to 0.05.
            verbose (_type_, optional): _description_. Defaults to True.
            graph (_type_, optional): _description_. Defaults to True.
            step (_type_, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        steps = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
        if graph:
            plt.plot(steps, costs, 'b-')
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
        if filename[-4:] != '.pkl':
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

    @staticmethod
    def sigmoid(x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1 / (1 + np.exp(-x))
