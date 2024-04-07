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
    pi = np.full((k,), 1 / k)
    m = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))
    S = np.full((k, d, d), np.identity(d))
    return pi, m, S
