#!/usr/bin/env python3
"""_summary_
"""

import numpy as np
"""_summary_

    Raises:
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_
"""


class NeuralNetwork:
    """_summary_
    """
    def __init__(self, nx, nodes):
        """_summary_

        Args:
            nx (_type_): _description_
            nodes (_type_): _description_
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0

    @property
    def W1(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__W1

    @property
    def b1(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__b1

    @property
    def A1(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__A1

    @property
    def W2(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__W2

    @property
    def b2(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__b2

    @property
    def A2(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__A2
