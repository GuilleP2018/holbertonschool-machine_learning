#!/usr/bin/env python3
""" Function that performs matrix multiplication"""
import numpy as np


def np_matmul(mat1, mat2):
    """Use Numpy to multiply matrixes

    Args:
        mat1: matrix 1
        mat2: matrix 2
    Returns:
        numpy.ndarray that is the multiplication of the matrixes
    """
    return np.matmul(mat1, mat2)
