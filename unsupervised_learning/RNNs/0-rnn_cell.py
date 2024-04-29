#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


class RNNCell:
    """_summary_
    """

    def __init__(self, i, h, o):
        """_summary_

        Args:
            i (_type_): _description_
            h (_type_): _description_
            o (_type_): _description_
        """
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

    def forward(self, h_prev, x_t):
        m, i = np.shape(x_t)
        m, h = np.shape(h_prev)
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
