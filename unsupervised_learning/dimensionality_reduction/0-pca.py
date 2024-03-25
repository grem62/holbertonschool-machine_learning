#!/usr/bin/env python3
"""_summary_
"""


import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X (np.ndarray): Dataset of shape (n, d).
        var (float): Fraction of variance to maintain.

    Returns:
        np.ndarray: Weights matrix W of shape (d, nd).
    """
    n, d = X.shape
    X_m = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X_m)
    total_var = np.sum(S)
    var_exp = np.cumsum(S) / total_var
    ndim = np.argmax(var_exp >= var) + 1
    W = V.T[:, :ndim]
    return W
