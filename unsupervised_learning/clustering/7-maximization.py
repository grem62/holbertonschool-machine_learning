#!/usr/bin/env python3

import numpy as np


def maximization(X, g):
    # X: Input data matrix of shape (n, d),
    # where n is the number of samples and d is the number of features
    # g: Matrix of shape (k, n) representing the
    # posterior probabilities of each sample belonging to each cluster
    n, d = X.shape
    k = g.shape[0]

    # Calculate updated mixture coefficients (pi) values
    pi = np.sum(g, axis=1) / n

    # Initialize arrays to store updated means (m) and covariances (S)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    # Update means and covariances for each cluster
    for i in range(k):
        # Calculate updated mean for cluster i
        m[i] = np.dot(g[i], X) / np.sum(g[i])

        # Calculate the difference between each sample
        # and the mean of cluster i
        diff = X - m[i]

        # Calculate updated covariance matrix for cluster i
        S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    # Return the updated mixture coefficients, means, and covariances
    return pi, m, S
