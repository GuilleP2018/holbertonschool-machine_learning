#!/usr/bin/env python3
"""This modlue contains the RNNCell class"""

import numpy as np


class RNNCell:
    """This class reperesents a cell of a simple Rnn"""

    def __init__(self, i, h, o):
        """This is the class constructor
        """
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """This methond preforms forward pass for one time step
        """
        input_data = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(input_data, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    def softmax(self, y):
        """This method calculates the softmax activation function
        """
        return np.exp(y) / (np.sum(np.exp(y), axis=1, keepdims=True))
