#!/usr/bin/env python3
"""_summary_"""


import numpy as np
"""_summary_"""


def batch_norm(Z, gamma, beta, epsilon):
    """_summary_

    Args:
        Z (_type_): _description_
        gamma (_type_): _description_
        beta (_type_): _description_
        epsilon (_type_): _description_

    Returns:
        _type_: _description_
    """
    m = np.mean(Z, axis=0)
    s = np.var(Z, axis=0)
    Z_norm = (Z - m) / np.sqrt(s + epsilon)
    return gamma * Z_norm + beta
