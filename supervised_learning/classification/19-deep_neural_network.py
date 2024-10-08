#!/usr/bin/env python3
"""This module creates a deep neural network class"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class"""
    def __init__(self, nx, layers):
        """Class constructor for DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2/nx)
            else:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2/layers[i - 1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Function that returns L"""
        return self.__L

    @property
    def cache(self):
        """Function that returns cache"""
        return self.__cache

    @property
    def weights(self):
        """Function that returns weights"""
        return self.__weights

    def forward_prop(self, X):
        """Function that calculates the forward propagation
        of the neural network"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            W = self.weights["W" + str(i + 1)]
            b = self.weights["b" + str(i + 1)]
            A = self.cache["A" + str(i)]
            Z = np.matmul(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.cache["A" + str(i + 1)] = A
        return A, self.cache

    def cost(self, Y, A):
        """Function that calculates the cost of the model using
        logistic regression"""
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost
