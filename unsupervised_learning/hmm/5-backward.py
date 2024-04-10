#!/usr/bin/env python3
"""_summary_"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Calculate the backward algorithm for
    Hidden Markov Models (HMM).

    Args:
        Observation (numpy.ndarray): Array of shape (T,)
        containing the index of the observation.
        Emission (numpy.ndarray): Array of shape (N, M)
        containing the emission probability of a specific
        observation given a hidden state.
        Transition (numpy.ndarray): Array of shape (N, N)
        containing the transition probabilities.
        Initial (numpy.ndarray): Array of shape (N, 1)
        containing the initial probabilities.

    Returns:
        tuple: A tuple containing the probability of
        the observations given the model and the backward probabilities.

    """
    Ob = Observation.shape[0]
    N = Transition.shape[0]
    B = np.zeros((N, Ob))
    B[:, Ob - 1] = 1
    for t in range(Ob - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(B[:, t + 1] *
                             Transition[i, :] *
                             Emission[:, Observation[t + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
