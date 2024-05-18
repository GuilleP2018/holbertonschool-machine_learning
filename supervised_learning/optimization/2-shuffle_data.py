#!/usr/bin/env python3
"""This module contains the function for shuffling data
"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
