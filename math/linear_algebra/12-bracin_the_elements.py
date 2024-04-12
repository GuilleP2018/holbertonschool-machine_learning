#!/usr/bin/env python3
"""Function that performs element-wise addition, subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """element-wise basic matrix calculator

    Args:
        mat1: first matrix, a numpy.ndarray
        mat2: second matrix, a numpy.ndarray
    Returns:
        the result of the element-wise matrix calculations
    """
    sum = mat1 + mat2
    difference = mat1 - mat2
    product = mat1 * mat2
    quotient = mat1 / mat2

    return (sum, difference, product, quotient)
