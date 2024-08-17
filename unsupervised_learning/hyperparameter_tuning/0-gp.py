#!/usr/bin/env python3
"""This module contains the function gp that performs the Gaussian Process"""
import numpy as np


class GaussianProcess:
    """Gaussian process class"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Init function for GaussianProcess"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, Y_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices"""
        sqdist = np.sum(X1**2, 1).reshape(
            -1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
