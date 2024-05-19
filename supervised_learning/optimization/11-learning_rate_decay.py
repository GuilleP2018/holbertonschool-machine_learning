#!/usr/bin/env python3
"""This module contains the function for updating a variable
using the Adam
optimization algorithm
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Update the learning rate using inverse time decay in numpy.
    """

    # Calculate the decay factor
    decay_factor = 1 / (1 + decay_rate * np.floor(global_step / decay_step))

    # Update the learning rate
    alpha_updated = alpha * decay_factor

    # Return the updated learning rate
    return alpha_updated
