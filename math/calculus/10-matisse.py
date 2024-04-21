#!/usr/bin/env python3
"""Function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial"""
    # Check if poly is a list or is empty
    if type(poly) is not list or len(poly) == 0:
        return None
    # Check if poly is a list of integers
    if len(poly) == 1:
        return [0]
    # Return the derivative of the polynomial
    return [poly[i] * i for i in range(1, len(poly))]
