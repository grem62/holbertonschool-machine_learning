#!/usr/bin/env python3

"""_summary_
"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """Initialization of a neuron

        Args:
            nx (int): number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter function for private instance W

        Returns:
            int: Weights vector for the neuron
        """
        return self.__W

    @property
    def b(self):
        """Getter function for private instance b

        Returns:
            int: bias for the neuron
        """
        return self.__b

    @property
    def A(self):
        """Getter function for private instance A

        Returns:
            int: Activated output of the neuron (prediction)
        """
        return self.__A

    def forward_prop(self, X):
        """_summary_

        Args:
            X (_type_): _description_
        """
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A
        """functionfor calculate the forward propagation of the neuron"""

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
        """ mathematic function for calculate the cost of model"""
        return cost

    def evaluate(self, X, Y):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.round(A).astype(int), cost
        """function for evaluate the neuron"""

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            A (_type_): _description_
            alpha (float, optional): _description_. Defaults to 0.05.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
        """function for calculate the gradient descent for neuron"""

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            iterations (int, optional): _description_. Defaults to 5000.
            alpha (float, optional): _description_. Defaults to 0.05.

        Returns:
            _type_: _description_
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
        return self.evaluate(X, Y)
