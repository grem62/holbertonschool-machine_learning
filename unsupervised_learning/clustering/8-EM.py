#!/usr/bin/env python3
""" Expectation Maximization for Gaussian Mixture Model """

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X,
                             num_clusters,
                             iterations=1000,
                             tolerance=1e-5,
                             verbose=False):
    """
    Performs Expectation Maximization for Gaussian Mixture Model
    Args:
        - X: np.ndarray (n, d) - data set
            - n: number of data points
            - d: number of dimensions
        - num_clusters: positive int
        - number of clusters
        - iterations: positive int - number of iterations
        - tolerance: non-negative float - tolerance of log likelihood
        - verbose: bool - print information
    Returns:
        - pi: np.ndarray (num_clusters,) - priors for each cluster
        - means: np.ndarray (num_clusters, d) - centroid means for each cluster
        - covariances: np.ndarray (num_clusters, d, d)
        -covariance matrices for each cluster
        - probabilities: np.ndarray (num_clusters, n)
        - probabilities for each data point in each cluster
        - log_likelihood: float - log likelihood of the model
    """
    # Input Validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(num_clusters, int) or num_clusters <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tolerance, float) or tolerance <= 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Initialization
    priors, means, covariances = initialize(X, num_clusters)
    probabilities, log_likelihood = expectation(X, priors, means, covariances)

    # Store the previous log likelihood
    prev_log_likelihood = 0

    # EM iterations
    for i in range(iterations):
        # Verbose mode: print log likelihood after every 10 iterations
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(
                i, log_likelihood.round(5)))

        # Maximization step
        priors, means, covariances = maximization(X, probabilities)

        # Expectation step
        probabilities, log_likelihood = expectation(
            X, priors, means, covariances)

        # Check convergence
        if np.abs(log_likelihood - prev_log_likelihood) <= tolerance:
            break

        # Update previous log likelihood
        prev_log_likelihood = log_likelihood

    # Final log likelihood
    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            i+1, log_likelihood.round(5)))

    return priors, means, covariances, probabilities, log_likelihood
