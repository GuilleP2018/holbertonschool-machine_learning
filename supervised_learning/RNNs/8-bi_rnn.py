#!/usr/bin/env python3
"""This moduel contiones the bi_rnn function"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """This function performs forward propagation for a bidirectional RNN
    """
    t, m, i = X.shape
    _, h = h_0.shape
    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))
    Hf[0] = h_0
    Hb[-1] = h_t

    for i in range(t):
        Hf[i + 1] = bi_cell.forward(Hf[i], X[i])

    for i in range(t - 1, -1, -1):
        Hb[i] = bi_cell.backward(Hb[i + 1], X[i])

    H = np.concatenate((Hf[1:], Hb[:t]), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
