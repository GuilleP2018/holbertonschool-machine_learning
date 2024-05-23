#!/usr/bin/env python3
"""Create confusion module """

import numpy as np


def create_confusion_matrix(labels, logits):
    """Create confusion matrix function"""

    return np.dot(labels.T, logits)
