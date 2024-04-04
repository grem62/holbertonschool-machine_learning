#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """_summary_

    Args:
        X (_type_): _description_
        k (_type_): _description_
        iterations (int, optional): _description_. Defaults to 1000.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    
    # Détermination du nombre de lignes et de colonnes de la matrice X
    n, d = X.shape

    # Vérification que k est un entier positif
    if not isinstance(k, int) or k <= 0:
        return None, None

    # Vérification que iterations est un entier positif
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialisation des centres de cluster de manière aléatoire
    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))

    # Boucle pour effectuer les itérations du K-means
    for _ in range(iterations):
        # Copie des centres de cluster actuels
        C_copy = np.copy(C)

        # Calcul de la distance entre chaque point et chaque centre de cluster
        D = np.linalg.norm(X - C[:, np.newaxis], axis=2)

        # Attribution de chaque point au centre de cluster le plus proche
        clss = np.argmin(D, axis=0)

        # Mise à jour des centres de cluster
        for j in range(k):
            # Si un cluster est vide, un nouveau centre est généré
            if len(X[clss == j]) == 0:
                C[j] = np.random.uniform(np.min(X, axis=0),
                                         np.max(X, axis=0), (1, d))
        # Sinon, le centre est mis à jour en prenant la moyenne des points du cluster
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        # Vérification de la convergence du K-means
        if np.array_equal(C, C_copy):
            return C, clss
    
    D = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    clss = np.argmin(D, axis=0)
    return C, clss