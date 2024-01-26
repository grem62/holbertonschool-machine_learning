!/usr/bin/env python3
"""_summary_
"""


import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function that updates the weights of a neural network
    with Dropout regularization using gradient descent

    Args:
       Y: is a one-hot numpy.ndarray of shape (classes, m)
       that contains the correct labels for the data
          classes: is the number of classes
          m: is the number of data points
       weights: is a dictionary of the weights and biases of the neural network
       cache: is a dictionary of the outputs and dropout
       masks of each layer of the neural network
       alpha: is the learning rate
       keep_prob: is the probability that a node will be kept
       L: is the number of layers of the network
    """
    N = Y.shape[1]

    dZ_curr = cache['A' + str(L)] - Y

    for l_n in range(L, 0, -1):
        c_W_n = 'W' + str(l_n)
        curr_W = weights[c_W_n]
        p_A_n = 'A' + str(l_n - 1)
        prev_A = cache[p_A_n]

        dW_curr = 1 / N * np.matmul(dZ_curr, prev_A.T)
        dB_curr = 1 / N * np.sum(dZ_curr, axis=1, keepdims=True)

        if l_n > 1:
            dZ_curr = np.matmul(curr_W.T, dZ_curr)
            dZ_curr *= cache['D' + str(l_n - 1)] / keep_prob
            dZ_curr *= 1 - prev_A**2
        else:
            dZ_curr = np.matmul(curr_W.T, dZ_curr) * (1 - prev_A**2)

        weights[c_W_n] -= alpha * dW_curr
        weights['b' + str(l_n)] -= alpha * dB_curr
