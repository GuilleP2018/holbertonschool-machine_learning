#!/usr/bin/env python3
""" Module for matrix_transpose method """


def matrix_transpose(matrix):
    """ Returns the transpose of a 2D matrix
    Args:
        matrix: a 2D matrix
        Returns:
        a new 2D matrix with the rows and columns flipped
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
