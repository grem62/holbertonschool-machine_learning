#!/usr/bin/env python3
"""
_summary_
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
