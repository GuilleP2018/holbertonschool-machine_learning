#!/usr/bin/env python3
"""This module contains the function for updating a variable using the Adam
optimization algorithm
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update a variable in place using the Adam optimization algorithm.
    """
    # Calculate the new first moment
    v_new = beta1 * v + (1 - beta1) * grad

    # Calculate the new second moment
    s_new = beta2 * s + (1 - beta2) * np.square(grad)

    # Correct the bias in the first moment
    v_corrected = v_new / (1 - np.power(beta1, t))

    # Correct the bias in the second moment
    s_corrected = s_new / (1 - np.power(beta2, t))

    # Update the variable
    var_updated = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    # Return the updated variable, the new
    # first moment, and the new second moment
    return var_updated, v_new, s_new
