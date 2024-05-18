#!/usr/bin/env python3
"""This module contains the function for normalizing a matrix
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalize the features in a matrix using provided
    mean and standard deviation.
    """

    # Subtract the mean and divide by the standard deviation
    X_normalized = (X - m) / s

    return X_normalized
