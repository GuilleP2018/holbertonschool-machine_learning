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

    def evaluate(self, X, Y):
        """Function that evaluates the neuron's predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Function that calculates one pass of gradient
        descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        db = np.sum(dz) / m
        dw = np.matmul(X, dz.T) / m
        self.__b -= alpha * db
        self.__W -= alpha * dw.T

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Function that trains the neuron"""
        if isinstance(iterations, int) is False:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if isinstance(alpha, float) is False:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
