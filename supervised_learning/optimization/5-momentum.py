#!/usr/bin/env python3
"""This module contains the function for updating
a variable using the gradient
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Update a variable using the gradient descent
    with momentum optimization algorithm.
    """
    # Calculate the new moment by taking the weighted average
    # of the previous moment and the gradient
    v_new = beta1 * v + (1 - beta1) * grad

    # Update the variable by subtracting the learning rate
    # times the new moment from the variable
    var_updated = var - alpha * v_new

    # Return the updated variable and the new moment
    return var_updated, v_new
