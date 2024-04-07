#!/usr/bin/env python3
""" Expectation Maximization for Gaussian Mixture Model """

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, num_clusters,
                             num_iterations=1000,
                             tolerance=1e-5,
                             verbose=False):
    """
    Performs the Expectation Maximization for a Gaussian Mixture Model
    Args:
        - X: np.ndarray (n, d) data set
            - n: number of data points
            - d: number of dimensions
        - num_clusters: positive int, number of clusters
        - num_iterations: positive int, number of iterations
        - tolerance: non-negative float, tolerance of log likelihood
        - verbose: bool for printing information
    Returns:
        - pi: np.ndarray (num_clusters,) of priors for each clust
        - means: np.ndarray (num_clusters, d) of centroid means for each clust
        - covariances: np.ndarray (num_clusters, d, d) of
        - covariance matrices for each cluster
        - probabilities: np.ndarray (num_clusters, n) of
        - probabilities for each data point in each cluster
        - log_likelihood: log likelihood of the model
    """
    # Input Validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(num_clusters, int) or num_clusters <= 0:
        return None, None, None, None, None
    if not isinstance(num_iterations, int) or num_iterations <= 0:
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
    for iteration in range(num_iterations):
        # Verbose mode: printing log likelihood after every 10 iterations
        if verbose and iteration % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(
                iteration, log_likelihood.round(5)))

        # Maximization and Expectation steps
        priors, means, covariances = maximization(X, probabilities)
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
            iteration+1, log_likelihood.round(5)))

    return priors, means, covariances, probabilities, log_likelihood
