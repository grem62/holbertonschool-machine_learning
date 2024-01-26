#!/usr/bin/env python3
"""
This module contains :
A function that updates the weights of a neural network
with Dropout regularization using gradient descent

Function:
   def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):

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

    c_A_n = 'A{}'.format(L)
    curr_A = cache[c_A_n]

    dZ_curr = curr_A - Y

    for l_n in range(L, 0, -1):

        c_W_n = 'W{}'.format(l_n)
        curr_W = weights[c_W_n]
        p_A_n = 'A{}'.format(l_n - 1)
        prev_A = cache[p_A_n]

        dW_curr = 1 / N * np.matmul(dZ_curr, prev_A.T)
        dB_curr = 1 / N * np.sum(dZ_curr, axis=1, keepdims=True)

        next_dtanh = 1 - (np.power(prev_A, 2))

        dA = np.matmul(curr_W.T, dZ_curr) * next_dtanh

        if l_n - 1 > 0:
            D = cache['D{}'.format(l_n - 1)]
            dA = dA * D
            dA = dA / keep_prob

        dZ_curr = np.multiply(dA, next_dtanh)

        c_b_n = 'b{}'.format(l_n)

        weights[c_W_n] = weights[c_W_n] - alpha * dW_curr
        weights[c_b_n] = weights[c_b_n] - alpha * dB_curr
