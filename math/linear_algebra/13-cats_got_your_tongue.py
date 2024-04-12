#!/usr/bin/env python3
"""Concatenates two matrices along a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concats mat1 and mat2 along axis

    Args:
        mat1: matrix 1 for concat
        mat2: matrix 2 for concat
        axis: Axis along which to concatenate. Defaults to 0.
    Returns:
        concatenated numpy.ndarray
    """
    return np.concatenate((mat1, mat2), axis=axis)
