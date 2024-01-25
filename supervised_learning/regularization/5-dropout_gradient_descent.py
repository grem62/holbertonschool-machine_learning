#!/usr/bin/env python3
"""_summary_"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """_summary_

    Args:
        Y (_type_): _description_
        weights (_type_): _description_
        cache (_type_): _description_
        alpha (_type_): _description_
        keep_prob (_type_): _description_
        L (_type_): _description_
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        dW = np.matmul(dZ, cache['A' + str(i - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = np.matmul(weights['W' + str(i)].T, dZ)
        if i > 1:
            dZ = np.multiply(dZ, cache['A' + str(i - 1)])
            dZ = np.multiply(dZ, 1 - cache['A' + str(i - 1)])
            dZ /= keep_prob
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db