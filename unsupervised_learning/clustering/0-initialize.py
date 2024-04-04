#!/usr/bin/env python3

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means.

    Args:
        X (np.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        np.ndarray: Centroids of shape (k, d).
    """
    n, d = X.shape
    min_X = np.min(X, axis=0)
    max_X = np.max(X, axis=0)
    return np.random.uniform(min_X, max_X, (k, d)) or None
