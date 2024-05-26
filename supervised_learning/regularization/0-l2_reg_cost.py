#!/usr/bin/env python3
"""L2 Regularization Cost module"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization"""
    norm = 0
    for i in range(1, L + 1):
        norm += np.linalg.norm(weights['W' + str(i)])
    return cost + (norm * lambtha / (2 * m))
