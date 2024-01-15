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
    shuff = np.random.permutation(X.shape[0])
    return X[shuff], Y[shuff]
