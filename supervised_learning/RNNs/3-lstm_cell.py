#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


class LSTMCell:
    """_summary_
    """
    def __init__(self, i, h, o):
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """_summary_

        Args:
            h_prev (_type_): _description_
            c_prev (_type_): _description_
            x_t (_type_): _description_

        Returns:
            _type_: _description_
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = np.dot(concat, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))
        u = np.dot(concat, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))
        c_hat = np.dot(concat, self.Wc) + self.bc
        c_hat = np.tanh(c_hat)
        c_next = f * c_prev + u * c_hat
        o = np.dot(concat, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))
        h_next = o * np.tanh(c_next)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, c_next, y
