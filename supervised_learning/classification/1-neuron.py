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
