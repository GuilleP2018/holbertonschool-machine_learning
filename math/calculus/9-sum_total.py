#!/usr/bin/env python3
"""Write a function that calculates sum of i^2 for i from 1 to n"""


def summation_i_squared(n):
    """Funtion for calculating sum of i^2 for i from 1 to n"""
    if type(n) is not int or n < 1:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
