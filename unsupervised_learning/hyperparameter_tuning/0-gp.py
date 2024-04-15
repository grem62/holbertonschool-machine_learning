#!/usr/bin/env python3
"""Gaussian Process for hyperparameter tuning."""

import numpy as np


class GaussianProcess:
    """
    class Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """_summary_

        Args:
            X_init (_type_): _description_
            Y_init (_type_): _description_
            l (int, optional): _description_. Defaults to 1.
            sigma_f (int, optional): _description_. Defaults to 1.
        """
        self.X = X_init
        self.Y = Y_init
        self.leight = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Compute the covariance kernel matrix
        using the Radial Basis Function (RBF).

        Args:
            X1 (numpy.ndarray): Input array of shape (m, 1).
            X2 (numpy.ndarray): Input array of shape (n, 1).

        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n).
        """
        return self.sigma_f**2 * np.exp(-0.5 / self.leight**2 * (X1 - X2.T)**2)
