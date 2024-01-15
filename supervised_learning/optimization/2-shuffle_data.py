#!/usr/bin/env python3
"""
    _summary_
"""

import numpy as np
"""_summary_
"""


def shuffle_data(X, Y):
    """_summary_

    Args:
        X (_type_): _description_
        Y (_type_): _description_
    """
    Matrice1 = np.random.permutation(X)
    Matrice2 = np.random.permutation(Y)
    return Matrice1, Matrice2
