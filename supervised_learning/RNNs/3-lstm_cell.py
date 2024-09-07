#!/usr/bin/env python3
"""This module contains the LSTMCell class"""
import numpy as np


class LSTMCell:
    """This class represents an LSTM unit"""

    def __init__(self, i, h, o):
        """Class constructor
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))

        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))

        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))

        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """This function performs forward propagation for one time step
        """
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        f_gate = self.sigmoid(np.matmul(cell_input, self.Wf) + self.bf)
        u_gate = self.sigmoid(np.matmul(cell_input, self.Wu) + self.bu)
        c_intermidiary = np.tanh(np.matmul(cell_input, self.Wc) + self.bc)
        c_next = c_prev * f_gate + u_gate * c_intermidiary
        o_gate = self.sigmoid(np.matmul(cell_input, self.Wo) + self.bo)
        h_next = o_gate * np.tanh(c_next)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y

    def sigmoid(self, x):
        """This method calculates the sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """This method calculates the softmax function
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
