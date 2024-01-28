#!/usr/bin/env python3
"""_summary_"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """_summary_

    Args:
        X (_type_): _description_
        weights (_type_): _description_
        L (_type_): _description_
        keep_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.dot(W, A_prev) + b
        if i == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D
        cache['A' + str(i)] = A
    return cache
