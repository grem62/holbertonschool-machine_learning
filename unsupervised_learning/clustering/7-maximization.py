#!/usr/bin/env python3
""" Maximization step in the EM algorithm for a GMM """

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
        - X: np.ndarray (n, d) data set
            - n: number of data points
            - d: number of dimensions
        - g: np.ndarray (k, n) posterior probs for each data point in each
            cluster
    Returns: pi, mu, sigma, or None, None, None on failure
        - pi: np.ndarray (k,) updated priors for each clusterr
        - mu: np.ndarray (k, d) updated centroid means for each cluste
        - sigma: np.ndarray (k, d, d) updated covariance
        - matrices for each cluster
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    # Extracting dimensions
    n, d = X.shape

    # Further validation
    if g.shape[1] != n:
        return None, None, None
    # Number of clusters
    k = g.shape[0]
    if g.shape[0] != k:
        return None, None, None
    # Checking if posterior probabilities sum to 1
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    # Calculating updated priors, centroid means, and covariance matrices
    pi = np.zeros((k,))
    mu = np.zeros((k, d))
    sigma = np.zeros((k, d, d))

    for i in range(k):
        # Updating priors probabilities
        pi[i] = np.sum(g[i]) / n

        # Updating centroid means
        mu[i] = np.dot(g[i], X) / np.sum(g[i])

        # Updating covariance matrices
        diff = X - mu[i]
        sigma[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    # Return updated priors, centroid means, and covariance matrices
    return pi, mu, sigma
