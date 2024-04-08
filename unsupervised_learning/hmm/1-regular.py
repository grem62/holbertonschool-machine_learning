#!/usr/bin/env python3

import numpy as np


def regular(P):
    # Vérifier si P est une matrice carrée
    n, m = P.shape
    if n != m:
        return None

    # Vérifier si P est régulière
    is_regular = np.all(np.linalg.matrix_power(P, n) > 0)
    if not is_regular:
        return None

    # Initialiser le vecteur de probabilité
    prob_vector = np.ones((1, n)) / n

    # Itérer jusqu'à la convergence
    max_iterations = 1000
    tolerance = 1e-6
    for i in range(max_iterations):
        new_prob_vector = np.dot(prob_vector, P)
        if np.linalg.norm(new_prob_vector - prob_vector) < tolerance:
            return new_prob_vector
        prob_vector = new_prob_vector

    # Retourner None si la convergence n'est pas atteinte
    return None
