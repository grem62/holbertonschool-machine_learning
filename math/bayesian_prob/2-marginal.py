#!/usr/bin/env python3
"""marginal probability"""

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


def intersection(x, n, P, Pr):
    """_summary_

    Args:
        x (int): number of patients that develop side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array containing the various hypothetical
        probabilities of developing side effects
        Pr (numpy.ndarray): 1D array containing the prior beliefs of P

    Returns:
        numpy.ndarray: 1D array containing the intersection
        of obtaining x and n
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """_summary_

    Args:
        x (int): number of patients that develop side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D array containing the various hypothetical
        probabilities of developing side effects
        Pr (numpy.ndarray): 1D array containing the prior beliefs of P

    Returns:
        numpy.ndarray: 1D array containing the marginal probability
        of obtaining x and n
        with each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        e_msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(e_msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for value in P:
        if value < 0 or value > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for value in Pr:
        if value < 0 or value > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    return np.sum(intersection(x, n, P, Pr))
