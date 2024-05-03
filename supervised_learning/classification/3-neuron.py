#!/usr/bin/env python3
"""This module creates a neuron class"""
import numpy as np


class Neuron:
    """Neuron class"""
    def __init__(self, nx):
        """Function that initializes the data in the class"""
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Function that returns W"""
        return self.__W

    @property
    def b(self):
        """Function that returns b"""
        return self.__b

    @property
    def A(self):
        """Function that returns A"""
        return self.__A

    def forward_prop(self, X):
        """Function that calculates the forward propagation of the neuron"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Function that calculates the cost of the model
        Using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost
