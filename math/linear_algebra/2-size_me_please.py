#!/usr/bin/env python3
"""Function matrix_shape"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
