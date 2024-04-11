#!/usr/bin/env python3
""" Module for add_arrays method """



def add_arrays(arr1, arr2):
    """ Adds two arrays element-wise
    Args:
        arr1: a list of integers/floats
        arr2: a list of integers/floats
    Returns:
        a new list containing the element-wise sum of arr1 and arr2
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]