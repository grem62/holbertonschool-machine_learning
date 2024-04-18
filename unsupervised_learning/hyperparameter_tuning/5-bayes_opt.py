#!/usr/bin/env python3
"""Bayesian optimization on a noiseless 1D function"""

import numpy as np
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization on a noiseless 1D function"""

    def __init__(self,
                 f,
                 X_init,
                 Y_init,
                 bounds,
                 ac_samples,
                 l=1,
                 sigma_f=1,
                 xsi=0.01,
                 minimize=True):
        """Constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

    def acquisition(self):
        """_summary_"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            f_plus = np.min(self.gp.Y)
            imp = f_plus - mu - self.xsi
        else:
            f_plus = np.max(self.gp.Y)
            imp = mu - f_plus - self.xsi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """Optimizes the black-box function"""
        for _ in range(iterations):
            X_next, ei = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            Y_opt = self.gp.X[np.argmin(self.gp.Y)]
            X_opt = [np.min(self.gp.Y)]
        else:
            Y_opt = self.gp.X[np.argmax(self.gp.Y)]
            X_opt = [np.max(self.gp.Y)]
        self.gp.X = self.gp.X[:-1]
        X_opt_arrondi = np.round(X_opt, 8)
        return Y_opt, X_opt_arrondi
