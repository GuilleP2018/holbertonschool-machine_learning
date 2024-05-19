#!/usr/bin/env python3
"""This module normalized an unactivated output of a neural network
using batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalize an unactivated output of a neural
    network using batch normalization.
    """
    # Calculate the mean and variance of each feature
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    # Normalize the features
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)

    # Scale and shift the normalized features
    Z_out = gamma * Z_norm + beta

    return Z_out
