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

        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
