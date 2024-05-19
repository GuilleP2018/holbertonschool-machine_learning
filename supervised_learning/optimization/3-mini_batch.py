#!/usr/bin/env python3
"""This module contains the function for
training a mini-batch gradient
"""
import numpy as np


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches for training a neural
    network using mini-batch gradient descent.
    """
    m = X.shape[0]
    mini_batches = []

    # Shuffle X and Y
    shuffle_data = __import__("2-shuffle_data").shuffle_data
    X, Y = shuffle_data(X, Y)

    # Partition (X, Y). Minus the end case.
    num_complete_minibatches = (
        m // batch_size
    )  # number of mini batches of size batch_size
    for k in range(0, num_complete_minibatches):
        X_batch = X[k * batch_size: k * batch_size + batch_size, :]
        Y_batch = Y[k * batch_size: k * batch_size + batch_size, :]
        mini_batches.append((X_batch, Y_batch))

    # Handling the end case (last mini-batch < batch_size)
    if m % batch_size != 0:
        X_batch = X[num_complete_minibatches * batch_size: m, :]
        Y_batch = Y[num_complete_minibatches * batch_size: m, :]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
