#!/usr/bin/env python3
"""correlation"""

import numpy as np


def correlation(C):
    """
        calculates a correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = np.diag(C)
    d_sqrt = np.sqrt(d)
    outer = np.outer(d_sqrt, d_sqrt)
    correlation = C / outer

    return correlation
