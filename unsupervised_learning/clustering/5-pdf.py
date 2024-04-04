#!/usr/bin/env python3

import numpy as np


def pdf(X, m, S):
    """
    This function calculates the probability density function of a Gaussian
    distribution.

    Args:
        X (numpy.ndarray): The dataset.
        m (numpy.ndarray): The mean of the distribution.
        S (numpy.ndarray): The covariance matrix of the distribution.

    Returns:
        numpy.ndarray: The PDF values for each data point.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    n, d = X.shape
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    den = np.sqrt((2 * np.pi) ** d * det)
    X_m = X - m
    X_m = np.dot(X_m, inv)
    X_m = np.sum(X_m * (X - m), axis=1)
    pdf = np.exp(-X_m / 2) / den
    pdf = np.maximum(pdf, 1e-300)
    return pdf
