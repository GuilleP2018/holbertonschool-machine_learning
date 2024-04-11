#!/usr/bin/env python3
""" Module that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """ Adds two matrices element-wise
    Args:
        mat1: a list of lists of integers/floats
        mat2: a list of lists of integers/floats
    Returns:
        a new list of lists containing the element-wise sum of mat1 and mat2
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
