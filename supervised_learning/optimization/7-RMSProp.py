#!/usr/bin/env python3
"""This module contains the function for updating a variable using the RMSProp
optimization algorithm
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using the RMSProp optimization algorithm.
    """
    # Calculate the new moment by taking the weighted average
    # of the previous moment and the square of the gradient
    s_new = beta2 * s + (1 - beta2) * np.square(grad)

    # Update the variable by subtracting the learning
    # rate times the gradient
    # divided by the square root of the new moment plus
    # epsilon from the variable
    var_updated = var - alpha * grad / (np.sqrt(s_new) + epsilon)

    # Return the updated variable and the new moment
    return var_updated, s_new
