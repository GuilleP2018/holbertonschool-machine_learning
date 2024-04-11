#!/usr/bin/env python3
""" Concatenates two matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """concat mat1 with mat2 along axis

    Args:
        mat1 : 2D matrix containing ints/floats
        mat2 : 2D matrix containing ints/floats
        axis : Axis for the specific concatenation. Defaults to 0.
    Returns:
        Concatanated matrix or None if not posible to concatanate
    """
    if len(mat1[0]) != len(mat2[0]) and axis == 0:
        return None
    if len(mat1) != len(mat2) and axis == 1:
        return None
    concat = []
    if axis == 0:
        concat.extend(mat1)
        concat.extend(mat2)
    else:
        for i in range(len(mat1)):
            concat.append(mat1[i] + mat2[i])

    return concat
