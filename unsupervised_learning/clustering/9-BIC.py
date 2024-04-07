#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Calcule le BIC pour un ensemble de données donné et une plage de clusters.

    Args:
        X (numpy.ndarray): L'ensemble de données de forme (n, d).
        kmin (int): Le nombre minimum de clusters. Par défaut, 1.
        kmax (int): Le nombre maximum de clusters. S'il est None, il est défini
        sur le nombre de points de données dans X. Par défaut, None.
        iterations (int): Le nombre maximum d'itérations
        pour l'algorithme EM. Par défaut, 1000.
        tol (float): La tolérance pour la convergence de l'algorithme EM.
        Par défaut, 1e-5.
        verbose (bool): Si True, affiche des informations sur
        l'algorithme EM. Par défaut, False.

    Returns:
        best_k (int): Le nombre de clusters qui minimise le BIC.
        best_result (tuple): Valeurs optimales des paramètres
        (pi, m, S) pour le meilleur nombre de clusters.
        l (numpy.ndarray): Valeurs de log-vraisemblance
        pour chaque nombre de clusters.
        b (numpy.ndarray): Valeurs de BIC pour chaque nombre de clusters.
    """

    # Calcule le Critère d'Information Bayésien (BIC)
    # pour un ensemble de données donné et une plage de nombres de clusters

    expectation_maximization = __import__('8-EM').expectation_maximization

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= kmax:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax <= 0 or kmax > X.shape[0]:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    a = []
    b = []
    best_k = None
    best_result = None
    best_bic = np.inf

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        p = k * d + k * d * (d + 1) / 2 + (k - 1)
        bic = p * np.log(n) - 2 * ll

        a.append(ll)
        b.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    z = np.array(a)
    b = np.array(b)

    return best_k, best_result, z, b
