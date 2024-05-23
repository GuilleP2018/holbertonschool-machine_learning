#!/usr/bin/env python3
"""Specificity module"""

import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in a confusion matrix"""

    TN = np.sum(confusion) - (np.sum(confusion, axis=1) + np.sum(confusion, axis=0) - np.diag(confusion))
    FP = np.sum(confusion, axis=0) - np.diag(confusion)

    return TN / (TN + FP)
