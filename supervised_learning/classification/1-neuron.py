#!/usr/bin/env python3
"""
module numpy
"""
import numpy as np

"""
    class neuron that defines a single neuron performing binary classification
"""


class Neuron:
    """_summary_line
    """
    def __init__(self, nx):
        """_summary_

        Args:
            nx (_type_): _description_
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise TypeError("nx must be a positive integer")

        self.__W = np.random.normal(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """_summary_line"""
        return self.__W

    @property
    def b(self):
        """_summary_line"""
        return self.__b

    @property
    def A(self):
        """_summary_line"""
        return self.__A
