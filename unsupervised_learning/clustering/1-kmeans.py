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


def kmeans(X, k, iterations=1000):
    """This function performs K-means on a dataset"""

    centroids = initialize(X, k)

    if centroids is None:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    for iteration in range(iterations):
        prev_centroids = np.copy(centroids)

        X_vector = np.repeat(X, k, axis=0)

        X_vector = X_vector.reshape(n, k, d)

        centroids_vector = np.tile(centroids, (n, 1))

        centroids_vector = centroids_vector.reshape(n, k, d)

        distance = np.linalg.norm(X_vector - centroids_vector, axis=2)

        clss = np.argmin(distance ** 2, axis=1)

        for i in range(k):

            indixes = np.where(clss == i)[0]
            if len(indixes) == 0:
                centroids[i] = initialize(X, 1)
            else:
                centroids[i] = np.mean(X[indixes], axis=0)

        if np.all(prev_centroids == centroids):
            return centroids, clss

        centroids_vector = np.tile(centroids, (n, 1))
        centroids_vector = centroids_vector.reshape(n, k, d)
        distance = np.linalg.norm(X_vector - centroids_vector, axis=2)
        clss = np.argmin(distance ** 2, axis=1)

    return centroids, clss
