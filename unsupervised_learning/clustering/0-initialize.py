#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.

    Args:
        X (np.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        np.ndarray: Centroids of shape (k, d).
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None

    n, d = X.shape
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    return np.random.uniform(min_X, max_X, (k, d))
