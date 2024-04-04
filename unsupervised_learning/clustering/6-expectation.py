#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def expectation(X, pi, m, S):
    """
    This function calculates the expectation step
    in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): The dataset.
        pi (numpy.ndarray): The priors for each cluster.
        m (numpy.ndarray): The mean of the distribution.
        S (numpy.ndarray): The covariance matrix of the distribution.

    Returns:
        numpy.ndarray: The posterior probability
        for each data point in each cluster.
        numpy.ndarray: The total log likelihood.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]:
        return None, None
    if X.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None
    if np.abs(np.sum(pi) - 1) > 1e-5:
        return None, None
    if np.abs(np.sum(pi) - 1) > 1e-5:
        return None, None
    if np.abs(np.sum(pi) - 1) > 1e-5:
        return None, None
    pdf = __import__('5-pdf').pdf
    n, d = X.shape
    k = pi.shape[0]
    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    r = np.sum(np.log(np.sum(g, axis=0)))
    g = g / np.sum(g, axis=0)
    return g, r
