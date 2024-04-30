#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
    """


import numpy as np


class GRUCell:
    """Représente une cellule GRU pour un RN."""
    def __init__(self, i, h, o):
        """
        Initialisation de la cellule GRU.

        Args:
            i (int): Dimension de l'entrée.
            h (int): Dimension de l'état caché.
            o (int): Dimension de la sortie.
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Effectue une propagation avant à travers la cellule GRU.

        Args:
            h_prev (ndarray): État caché précédent.
            x_t (ndarray): Entrée à l'instant t.

        Returns:
            ndarray: Prochain état caché.
            ndarray: Sortie à l'instant t.
        """
        m, i = np.shape(x_t)
        m, h = np.shape(h_prev)
        x = np.hstack((h_prev, x_t))
        z = 1 / (1 + np.exp(-(np.dot(x, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.dot(x, self.Wr) + self.br)))
        x = np.hstack((r * h_prev, x_t))
        h_tilde = np.tanh(np.dot(x, self.Wh) + self.bh)
        h_next = z * h_tilde + (1 - z) * h_prev
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
