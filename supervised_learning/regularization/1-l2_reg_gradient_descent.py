#!/usr/bin/env python3
"""_summary_"""

import numpy as np
"""_summary_"""


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """_summary_

    Args:
        Y (_type_): _description_
        weights (_type_): _description_
        cache (_type_): _description_
        alpha (_type_): _description_
        lambtha (_type_): _description_
        L (_type_): _description_
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dW = (1 / m) * np.matmul(dZ, A.T) + ((lambtha / m) * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.matmul(W.T, dZ) * (1 - A * A)
        weights['W' + str(i)] = weights['W' + str(i)] - (alpha * dW)
        weights['b' + str(i)] = weights['b' + str(i)] - (alpha * db)
