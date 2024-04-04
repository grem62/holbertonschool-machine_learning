#!/usr/bin/env python3

import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set.
    # Calcule la variance intra-cluster totale pour un ensemble de données.

    Args:
        X (np.ndarray): Dataset of shape (n, d).
        # X (np.ndarray): Ensemble de données de forme (n, d).
        C (np.ndarray): Centroids of shape (k, d).
        # C (np.ndarray): Centroides de forme (k, d).

    Returns:
        float: Total variance.
        # float: Variance totale.
    """
    n, d = X.shape
    # Récupère le nombre de lignes et de colonnes de X
    k, d2 = C.shape
    # Récupère le nombre de lignes et de colonnes de C
    if d != d2:
        return None
        # Si le nombre de colonnes de X est différent
        # du nombre de colonnes de C, retourne None
    dist = np.linalg.norm(X[:, None] - C, axis=-1)
    # Calcule la distance entre chaque point de X et chaque centroid de C
    min_dist = np.min(dist, axis=-1)
    # Calcule la distance minimale entre chaque point de
    # X et les centroides de C
    return np.sum(min_dist ** 2)
    # Retourne la somme des distances minimales au carré
