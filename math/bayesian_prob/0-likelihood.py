#!/usr/bin/env python3
"""likelihood"""

import numpy as np


def likelihood(x, n, P):
    """_summary_

    Args:
        x (int): number of patients that develop side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array containing the various hypothetical
        probabilities of developing side effects

    Returns:
        numpy.ndarray: 1D array containing the likelihood of obtaining x and n
        with each probability in P, respectively
    """
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be an integer that is greater than 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    fact = np.math.factorial
    comb = fact(n) / (fact(x) * fact(n - x))
    likelihood = comb * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
