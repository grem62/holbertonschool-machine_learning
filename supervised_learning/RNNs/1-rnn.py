#!/usr/bin/env python3
"""Effectue une propagation avant pour un réseau de neurones récurrents."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Effectue une propagation avant pour un réseau de neurones récurrents.

    Args:
        rnn_cell (RNNCell): Cellule RNN utilisée.
        X (ndarray): Tableau de forme (t, m, i) contenant les données
                     d'entrée.
            t (int): Nombre d'instants dans la séquence.
            m (int): Nombre d'éléments dans le lot.
            i (int): Dimension de l'entrée.
        h_0 (ndarray): Tableau de forme (m, h) contenant l'état initial.

    Returns:
        ndarray: Tableau de forme (t, m, h) contenant les états cachés pour
                 chaque instant dans la séquence.
        ndarray: Tableau de forme (t, m, o) contenant les sorties pour chaque
                 instant dans la séquence.
    """
    t, m, i = np.shape(X)
    h = np.shape(h_0)[1]
    o = np.shape(rnn_cell.Wy)[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y[step] = y
    return H, Y
