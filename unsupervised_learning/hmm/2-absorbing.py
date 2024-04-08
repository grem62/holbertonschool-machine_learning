#!/usr/bin/env python3
"""abosrbing chain in markov"""

import numpy as np


def absorbing(P: np.ndarray) -> bool:
    """
    Check if a given matrix is an absorbing Markov chain.

    Args:
        P (np.ndarray): The transition probability matrix.

    Returns:
        bool: True if the matrix is an absorbing Markov chain, False otherwise.
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None
    for i in range(n):
        if np.all(P[i] == np.eye(n)[i]):
            return True
    return False
