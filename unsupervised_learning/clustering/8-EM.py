#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Effectue l'espérance-maximisation pour un GMM."""
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None
    # Initialiser les paramètres
    pi, m, S = initialize(X, k)

    # Initialiser la vraisemblance logarithmique
    prev_l = 0

    for i in range(iterations):
        # Étape d'espérance
        g, z = expectation(X, pi, m, S)

        # Étape de maximisation
        pi, m, S = maximization(X, g)

        # Calculer la différence de vraisemblance logarithmique
        diff = abs(z - prev_l)

        # Afficher la vraisemblance logarithmique si verbose est True
        if verbose and i % 10 == 0:
            print(f"Vraisemblance logarithmique après {i} itérations: {z:.5f}")

        # Vérifier la convergence
        if diff <= tol:
            break

        # Mettre à jour la vraisemblance logarithmique précédente
        prev_l = z

    if verbose:
        print(f"Vraisemblance logarithmique après {i} itérations: {z:.5f}")

    return pi, m, S, g, z
