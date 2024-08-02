#!/usr/bin/env python3
"""This module calculates the total intra-cluster
variance for a data set"""
import numpy as np


def variance(X, C):
    """This function calculates the total intra-cluster
    variance fr a data set
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing the centroid
        means fr each cluster
            - n is the number of data points
            - d is the number of dimensions fr each data point
            - k is the number of clusters
        Returns: total variance
                None: on failure
    """
    # Verification of data
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    # Process of the function
    n, d = X.shape
    k = C.shape[0]

    # Verification of Data
    if k > n:
        return None
    if d != C.shape[1]:
        return None

    #process of the function
    vectorized_data_X = np.repeat(X, k, axis=0)
    vectorized_data_X = vectorized_data_X.reshape(n, k, d)
    vectorized_centroids = np.tile(C, (n, 1))
    vectorized_centroids = vectorized_centroids.reshape(n, k, d)
    distance = np.linalg.norm(
        vectorized_data_X - vectorized_centroids, axis=2)
    dist_short = np.min(distance ** 2, axis=1)
    variance = np.sum(dist_short)

    return variance
