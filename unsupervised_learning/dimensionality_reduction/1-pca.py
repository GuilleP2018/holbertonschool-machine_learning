#!/usr/bin/env python3
"""
0-pca.py
"""
import numpy as np


def pca(X, ndim):
    """function that performs principal components analysis on a dataset"""

    X = X - np.mean(X, axis=0)

    # Compute the SVD:
    U, S, Vt = np.linalg.svd(X)

    # Compute the weights matrix
    W = Vt[:ndim].T

    # Transform the data
    T = np.matmul(X, W)

    return T

