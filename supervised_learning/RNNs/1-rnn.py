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
    t, m, i = X.shape
    m, h = h_0.shape
    o = rnn_cell.Wy.shape[1]
    H = np.zeros((t, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for timestep in range(t):
        H[timestep + 1], Y[timestep] = rnn_cell.forward(H[timestep],
                                                        X[timestep])
    return H[1:], Y
