#!/usr/bin/env python3
"""
Simple stochastic policy function
"""

import numpy as np

def policy(matrix, weight):
    """
    Computes a stochastic policy by taking a weighted combination of the
    state and weight matrices and applying a softmax function.
    """
    weighted_states = (matrix @ weight)
    e_x = np.exp(weighted_states - np.max(weighted_states))
    return e_x / np.sum(e_x)
