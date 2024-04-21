#!/usr/bin/env python3
"""Function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    # Check if poly is a list or is empty
    if type(poly) is not list or len(poly) == 0:
        return None
    # Check if poly is a list of integers
    for i in poly:
        if type(i) is not int:
            return None
    # Return the integral of the polynomial
    integral = [C]
    for i in range(len(poly)):
        integral.append(poly[i] / (i + 1))
    return integral
