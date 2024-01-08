#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


class DeepNeuralNetwork:
    """_summary_
    """
    def __init__(self, nx, layers):
        """_summary_

        Args:
            nx (_type_): _description_
            layers (_type_): _description_

        Raises:
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
        """

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or not layers or layers[0] < 1:
            raise ValueError('layers must be a list of positive integers')

        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(1, self.L + 1):
            if layers[i - 1] < 1 or type(layers[i - 1]) is not int:
                raise TypeError('layers must be a list of positive integers')
            key_W = 'W' + str(i)
            key_b = 'b' + str(i)

            self.weights[key_W] = np.random.randn(layers[i - 1], nx) * \
                np.sqrt(2 / nx)
            self.weights[key_b] = np.zeros((layers[i - 1], 1))
            nx = layers[i - 1]
