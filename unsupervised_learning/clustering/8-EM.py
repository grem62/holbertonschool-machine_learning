#!/usr/bin/env python3
"""_summary_
"""

import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM."""
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None
    # Initialize the parameters
    pi, m, S = initialize(X, k)

    # Initialize the log likelihood
    prev_l = 0

    for i in range(iterations):
        # Expectation step
        g, z = expectation(X, pi, m, S)

        # Maximization step
        pi, m, S = maximization(X, g)

        # Compute the difference in log likelihood
        diff = abs(z - prev_l)

        # Print log likelihood if verbose is True
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {z:.5f}")

        # Check for convergence
        if diff <= tol:
            break

        # Update the previous log likelihood
        prev_l = z

    if verbose:
        print(f"Log Likelihood after {i} iterations: {z:.5f}")

    return pi, m, S, g, z
