#!/usr/bin/env python3

import numpy as np


def expectation(X, pi, m, S):
    """
    This function calculates the expectation step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): The dataset.
        pi (numpy.ndarray): The priors for each cluster.
        m (numpy.ndarray): The mean of the distribution.
        S (numpy.ndarray): The covariance matrix of the distribution.

    Returns:
        numpy.ndarray: The posterior probability for each data point in each cluster.
        numpy.ndarray: The total log likelihood.
    """
    pdf = __import__('5-pdf').pdf
    n, d = X.shape
    k = pi.shape[0]
    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    l = np.sum(np.log(np.sum(g, axis=0)))
    g = g / np.sum(g, axis=0)
    return g, l