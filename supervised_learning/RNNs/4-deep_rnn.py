#!/usr/bin/env python3
"""_summary_
    """

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """_summary_

    Args:
        rnn_cells (_type_): _description_
        X (_type_): _description_
        h_0 (_type_): _description_
    """
    # Get the dimensions of the input
    t, m, i = X.shape
    # Get the dimensions of the initial hidden state
    l, _, h = h_0.shape

    # Initialize the hidden state matrix
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    # Initialize the output list
    Y = []

    # Iterate over each time step
    for step in range(t):
        # Iterate over each layer
        for layer in range(l):
            # Get the previous hidden state
            if layer == 0:
                h_prev = X[step]
            else:
                h_prev = H[step, layer - 1]

            # Compute the next hidden state
            h_next = rnn_cells[layer].forward(h_prev, H[step, layer])
            H[step + 1, layer] = h_next
