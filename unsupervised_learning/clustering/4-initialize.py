#!/usr/bin/env python3
"""This modlue initializes variables fr a Gaussian
Mixture Model
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """This function initializes variables fr a Gaussian Mixture Model
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None

    pi = np.full((k,), fill_value=1/k)

    m, _ = kmeans(X, k)

    S = np.full((k, d, d), np.identity(d))

    return pi, m, S
