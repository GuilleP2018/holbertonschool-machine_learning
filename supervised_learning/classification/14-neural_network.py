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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.A1, self.A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.round(A2).astype(int)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient
        descent on the neural network"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W1 = self.W1 - (alpha * dw1)
        self.__b1 = self.b1 - (alpha * db1)
        self.__W2 = self.W2 - (alpha * dw2)
        self.__b2 = self.b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
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
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        return self.evaluate(X, Y)
