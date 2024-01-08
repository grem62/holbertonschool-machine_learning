#!/usr/bin/env python3
"""Neural Network
"""
import numpy as np


class NeuralNetwork:
    """_summary_
    """
    def __init__(self, nx, nodes):
        """Class contructor

        Args:
            nx (integer): number of input features
            nodes (integer): number of nodes in the hidden layer
        """

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter function for private attribute W1

        Returns:
            integer: weights vector for the hidden layer
        """

        return self.__W1

    @property
    def b1(self):
        """Getter function for private attribute b1

        Returns:
            integer: bias for the hidden layer
        """

        return self.__b1

    @property
    def A1(self):
        """Getter function for private attribute A1

        Returns:
            integer: activated output for the hidden layer
        """

        return self.__A1

    @property
    def W2(self):
        """Getter function for private attribute W2

        Returns:
            integer: weights vector for the ouput neuron
        """

        return self.__W2

    @property
    def b2(self):
        """Getter function for private attribute b2

        Returns:
            integer: bias for the output neuron
        """

        return self.__b2

    @property
    def A2(self):
        """Getter function for private attribute A2

        Returns:
            integer: activated output for the output neuron
        """

        return self.__A2

    def forward_prop(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.__A1 = 1 / (1 + np.exp(-(np.matmul(self.__W1, X) + self.__b1)))
        self.__A2 = 1 / \
            (1 + np.exp(-(np.matmul(self.__W2, self.__A1) + self.__b2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """_summary_

        Args:
            Y (_type_): _description_
            A (_type_): _description_

        Returns:
            _type_: _description_
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        A = np.where(self.__A2 >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            A1 (_type_): _description_
            A2 (_type_): _description_
            alpha (float, optional): _description_. Defaults to 0.05.
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
