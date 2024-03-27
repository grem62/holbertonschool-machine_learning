#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset.

    Args:
        X (np.ndarray): Dataset of shape (n, d).
        ndim (int): Number of dimensions to reduce to.

    Returns:
        np.ndarray: Transformed dataset of shape (n, ndim).
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X_centered)
    T = np.dot(X_centered, V.T[:, :ndim])

    return T
