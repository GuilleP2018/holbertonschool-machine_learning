#!/usr/bin/env python3
""" Function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Multiplies mat1 and mat2

    Args:
        mat1: 2d matrix containing ints/floats
        mat2: 2D matrix containing ints/floats
    Returns:
        returns multiplied matrix or none in the case it wasnt possible
    """
    if len(mat1[0]) != len(mat2):
        return None
    new_mat = [[0 for col in range(len(mat2[0]))] for row in range(len(mat1))]

    for row1 in range(len(mat1)):
        for col2 in range(len(mat2[0])):
            for col1 in range(len(mat1[0])):
                new_mat[row1][col2] += mat1[row1][col1] * mat2[col1][col2]
    return new_mat
