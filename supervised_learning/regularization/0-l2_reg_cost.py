#!/usr/bin/env python3
"""_summary_
"""


import numpy as np
"""_summary_"""


def l2_reg_cost(cost, lambtha, weights, L, m):
    """_summary_

    Args:
        cost (_type_): _description_
        lambtha (_type_): _description_
        weights (_type_): _description_
        L (_type_): _description_
        m (_type_): _description_
    """
    new_cost = 0
    for i in range(1, L + 1):
        new_cost += np.linalg.norm(weights['W' + str(i)])
    return cost + (lambtha / (2 * m)) * new_cost
