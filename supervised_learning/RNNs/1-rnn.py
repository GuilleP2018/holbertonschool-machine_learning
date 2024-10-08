#!/usr/bin/env python3
"""This module contains the rnn function"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """This function preforms a forward propagation on a simple RNN
    """
    t = X.shape[0]
    m = X.shape[1]
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    for i in range(t):
        if i == 0:
            H[i] = h_0
        H[i + 1], Y[i] = rnn_cell.forward(H[i], X[i])

    return H, Y
