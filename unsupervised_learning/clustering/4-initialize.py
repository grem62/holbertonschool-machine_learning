#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    This function initializes variables for a Gaussian Mixture Model.

    Args:
        X (numpy.ndarray): The dataset.
        k (int): The number of clusters.

    Returns:
        pi (numpy.ndarray): The priors for each cluster, initialized evenly.
        m (numpy.ndarray): The mean for each cluster, initialized randomly.
        S (numpy.ndarray): The covariance matrix for each cluster,
        initialized as identity matrices.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None
    n, d = X.shape
    # Initialize the priors for each cluster (pi) with equal probability
    priors = np.full(shape=(k,), fill_value=1/k)
    # Initialize the mean for each cluster (m) using K-means
    means = kmeans(X, k)[0]
    # Initialize the covariance matrices for each cluster
    # (S) as identity matrices
    covariances = np.full(shape=(k, d, d), fill_value=np.identity(d))
    return priors, means, covariances
