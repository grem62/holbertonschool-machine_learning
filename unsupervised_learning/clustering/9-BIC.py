#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Computes the BIC for a given dataset and range of clusters.

    Args:
        X (numpy.ndarray): The dataset of shape (n, d).
        kmin (int): The minimum number of clusters. Default is 1.
        kmax (int): The maximum number of clusters. If None, it is set to
        the number of data points in X. Default is None.
        iterations (int): The maximum number of iterations
        for the EM algorithm. Default is 1000.
        tol (float): The tolerance for convergence of the
        EM algorithm. Default is 1e-5.
        verbose (bool): If True, prints information about
        the EM algorithm. Default is False.

    Returns:
        best_k (int): The number of clusters that minimizes the BIC.
        best_result (tuple): Optimal values of the parameters
        (pi, m, S) for the best number of clusters.
        l (numpy.ndarray): Log-likelihood values
        for each number of clusters.
        b (numpy.ndarray): BIC values for each number of clusters.
    """

    # Comment explaining the purpose of the function
    # Compute the Bayesian Information Criterion (BIC)
    # for a given dataset and range of cluster numbers

    expectation_maximization = __import__('8-EM').expectation_maximization

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= kmax:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax <= 0 or kmax > X.shape[0]:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    a = []
    b = []
    best_k = None
    best_result = None
    best_bic = np.inf

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        p = k * d + k * d * (d + 1) / 2 + (k - 1)
        bic = p * np.log(n) - 2 * ll

        a.append(ll)
        b.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    z = np.array(a)
    b = np.array(b)

    return best_k, best_result, z, b
