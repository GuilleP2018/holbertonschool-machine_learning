#!/usr/bin/env python3
"""This module contains a BidirectionaCell class of an RNN"""
import numpy as np


class BidirectionalCell:
    """This class represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """Class constructor
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """This method calculates the forward propagation for one time step
        """
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(cell_input, self.Whf) + self.bhf)

        return h_next