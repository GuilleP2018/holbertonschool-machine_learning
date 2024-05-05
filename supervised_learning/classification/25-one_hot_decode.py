#!/usr/bin/env python3
"""This module creates a one-hot decode function"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    try:
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception as e:
        return None
