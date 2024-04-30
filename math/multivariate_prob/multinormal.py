#!/usr/bin/env python3

"""multinormal"""

import numpy as np


class MultiNormal:
    """Multinormal class"""

    def __init__(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1).reshape(-1, 1)
        self.cov = np.dot((data - self.mean),
                          (data - self.mean).T) / (data.shape[1] - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.mean.shape[0], 1):
            raise ValueError("x must have the shape ({}, 1)".format(
                self.mean.shape[0]))
        pdf = np.exp(np.dot((x - self.mean).T,
                            np.dot(np.linalg.inv(
                                self.cov), x - self.mean)) / -2) / np.sqrt(
                                    (2 * np.pi) ** self.mean.shape[0] *
                                    np.linalg.det(
                                        self.cov))

        return pdf[0][0]
