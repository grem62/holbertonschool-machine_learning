#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def maximization(X, g):
    # X: Matrice de données d'entrée de forme (n, d),
    # où n est le nombre d'échantillons et d est le nombre de caractéristiques
    # g: Matrice de forme (k, n) représentant les
    # probabilités a posteriori de chaque échantillon appartenant à chaque cluster
    n, d = X.shape
    k = g.shape[0]

    # Calculer les valeurs mises à jour des coefficients de mélange (pi)
    pi = np.sum(g, axis=1) / n

    # Initialiser les tableaux pour stocker les moyennes (m) et les covariances (S) mises à jour
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    # Mettre à jour les moyennes et les covariances pour chaque cluster
    for i in range(k):
        # Calculer la moyenne mise à jour pour le cluster i
        m[i] = np.dot(g[i], X) / np.sum(g[i])

        # Calculer la différence entre chaque échantillon
        # et la moyenne du cluster i
        diff = X - m[i]

        # Calculer la matrice de covariance mise à jour pour le cluster i
        S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    # Retourner les coefficients de mélange, les moyennes et les covariances mises à jour
    return pi, m, S
