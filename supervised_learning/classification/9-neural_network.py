#!/usr/bin/env python3
"""This module creates a neural network class"""
import numpy as np


class NeuralNetwork:
    """Neural network class"""
    def __init__(self, nx, nodes):
        """Class constructor for NeuralNetwork"""

        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if isinstance(nodes, int) is False:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Function that returns W1"""
        return self.__W1

    @property
    def b1(self):
        """Function that returns b1"""
        return self.__b1

    @property
    def A1(self):
        """Function that returns A1"""
        return self.__A1

    @property
    def W2(self):
        """Function that returns W2"""
        return self.__W2

    @property
    def b2(self):
        """Function that returns b2"""
        return self.__b2

    @property
    def A2(self):
        """Function that returns A2"""
        return self.__A2
