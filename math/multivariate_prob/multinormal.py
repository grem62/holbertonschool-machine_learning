#!/usr/bin/env python3

"""multinormal"""

import numpy as np


class MultiNormal:
    """Multinormal class"""

    def __init__(self, data):
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise TypeError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape(-1, 1)
        self.cov = np.dot((data - self.mean),
                          (data - self.mean).T) / (data.shape[1] - 1)
