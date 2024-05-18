#!/usr/bin/env python3
"""This module contains the function for calculating
the normalization constants of a matrix
"""
import numpy as np


def normalization_constants(X):
    """
    Calculate the mean and standard deviation of each feature in a matrix.
    """

    # Calculate the mean of each feature
    mean = np.mean(X, axis=0)
    # Calculate the squared differences from the mean
    squared_diffs = (X - mean) ** 2
    # Calculate the variance (mean of squared differences)
    variance = np.mean(squared_diffs, axis=0)
    # Standard deviation is the square root of variance
    std_dev = np.sqrt(variance)

    return mean, std_dev
