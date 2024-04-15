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
        self.l = l
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
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * (X1 - X2.T)**2)

    def predict(self, X_s):
        """_summary_
        Args:
            X_s (_type_): _description_
        """
        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu_s.reshape(-1), np.diag(cov_s)

    def update(self, X_new, Y_new):
        """_summary_

        Args:
            X_new (_type_): _description_
            Y_new (_type_): _description_
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
