#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


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
    n, d = X.shape
    pi = np.full((k,), 1 / k)
    m = np.random.uniform(X.min(axis=0), X.max(axis=0), (k, d))
    S = np.full((k, d, d), np.identity(d))

    return pi, m, S
