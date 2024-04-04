#!/usr/bin/env python3

import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    This function tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): The dataset.
        kmin (int, optional): The minimum number of
        clusters to check for (inclusive). Default is 1.
        kmax (int, optional): The maximum number of
        clusters to check for (inclusive). Default is None.
        iterations (int, optional): The maximum number
        of iterations for K-means. Default is 1000.

    Returns:
        results (list): Outputs of K-means for each cluster size.
        d_vars (list): Difference in variance from the
        smallest cluster size for each cluster size.
    """
    if kmax is None:
        kmax = X.shape[0]

    if kmin < 1 or kmax < kmin or iterations < 1:
        return None, None

    results = []
    d_vars = []

    try:
        kmeans = __import__('1-kmeans').kmeans
        variance = __import__('2-variance').variance

        _, var_ref = kmeans(X, 1, iterations)
        for k in range(kmin, kmax + 1):
            centroids, _ = kmeans(X, k, iterations)
            results.append((centroids, _))
            if k == kmin:
                var_min = variance(X, centroids)
            d_vars.append(var_min - variance(X, centroids))

    except Exception as e:
        print("An error occurred:", e)
        return None, None

    return results, d_vars
