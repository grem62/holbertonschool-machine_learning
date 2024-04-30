#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""


import numpy as np


class RNNCell:
    """Cette classe représente une cellule
    de réseau de neurones récurrents (RNN)."""

    def __init__(self, i, h, o):
        """
        Initialise une cellule RNN avec les dimensions spécifiées.

        Args:
            i (int): Dimension de l'entrée.
            h (int): Dimension de l'état caché (hidden state).
            o (int): Dimension de la sortie.
        """
        self.bh = np.zeros((1, h))  # Initialisation du biais de l'état caché
        self.by = np.zeros((1, o))  # Initialisation du biais de sortie
        self.Wh = np.random.randn(i + h, h)  # Initialisation des poids cachés
        self.Wy = np.random.randn(h, o)  # Initialisation des poids de sortie

    def forward(self, h_prev, x_t):
        """
        Effectue une propagation avant à travers la cellule RNN.

        Args:
            h_prev (ndarray): État caché précédent.
            x_t (ndarray): Entrée à l'instant t.

        Returns:
            ndarray: Prochain état caché.
            ndarray: Sortie à l'instant t.
        """
        m, i = np.shape(x_t)
        m, h = np.shape(h_prev)
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)),
                                self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by  # Calcul de la sortie
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
