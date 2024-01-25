#!/usr/env/bin python3
"""_summary_
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """_summary_

    Args:
        X (_type_): _description_
        weights (_type_): _description_
        L (_type_): _description_
        keep_prob (_type_): _description_
    """
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        Z = np.matmul(weights['W' + str(i)], cache['A' + str(i - 1)]) + \
            weights['b' + str(i)]
        if i == L:
            t = np.exp(Z)
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache['A' + str(i)] = np.tanh(Z)
            cache['D' + str(i)] = np.random.rand(
                cache['A' + str(i)].shape[0],
                cache['A' + str(i)].shape[1]) < keep_prob
            cache['A' + str(i)] = np.multiply(
                cache['A' + str(i)], cache['D' + str(i)])
            cache['A' + str(i)] /= keep_prob
    return cache
