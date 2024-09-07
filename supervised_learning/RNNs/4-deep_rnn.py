#!/usr/bin/env python3
"""This module contains the function that performs
forward propagation on a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """This function performs forward propagation on a deep RNN
    """
    t, m, i = X.shape
    h = h_0.shape[-1]
    length = len(rnn_cells)
    H = np.zeros((t + 1, length, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    for i in range(t):
        for j in range(length):
            if i == 0:
                H[i, j] = h_0[j]
            if j == 0:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i, j], X[i])
            else:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(
                    H[i, j], H[i + 1, j - 1])

    return H, Y
