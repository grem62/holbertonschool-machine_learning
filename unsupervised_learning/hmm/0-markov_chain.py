#!/usr/bin/env python3
"""Ce module contient une fonction pour calculer
la probabilité d'état d'une chaîne de Markov.

    Returns:
        numpy.ndarray: La probabilité d'état de la chaîne de Markov.
"""
import numpy as np


def markov_chain(P, s, t=1):
    """Calcule la probabilité d'état d'une chaîne de Markov.

    Args:
        P (numpy.ndarray): Matrice de transition de la chaîne de Markov.
        s (numpy.ndarray): Vecteur d'état initial.
        t (int, optionnel): Nombre d'étapes de transition. Par défaut, 1.

    Returns:
        numpy.ndarray: La probabilité d'état de la chaîne de Markov.
    """
    proba_transition = np.linalg.matrix_power(P, t)
    proba_state = np.dot(s, proba_transition)
    return proba_state
