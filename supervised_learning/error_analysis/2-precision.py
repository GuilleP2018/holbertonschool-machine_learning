#!/usr/bin/env python3
"""Precision module"""

import numpy as np


def precision(confusion):
    """Calculate the precision for each class in a confusion matrix"""

    return np.diag(confusion) / np.sum(confusion, axis=0)
