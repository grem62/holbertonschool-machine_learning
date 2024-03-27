#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def pca(X, ndim):
    """_summary_

    Args:
        X (_type_): _description_
        ndim (_type_): _description_

    Returns:
        _type_: _description_
    """
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top ndim eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :ndim]

    # Transform the data
    T = np.dot(X_centered, selected_eigenvectors)

    return T
