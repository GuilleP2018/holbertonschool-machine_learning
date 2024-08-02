#!/usr/bin/env python3
"""This modlue initializes a cluster centroids fr K-means"""
import numpy as np


def initialize(X, k):
    """This function initializes cluster centroids fr K-means"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Extract the shape of the dataset
    n, d = X.shape
    # XCheck fr inpuit data k
    if not isinstance(k, int) or k <= 0 or k > n:
        return None

    centroids = np.random.uniform(
        low=np.min(X, axis=0), high=np.max(X, axis=0), size=(k, d)
    )
    return centroids
