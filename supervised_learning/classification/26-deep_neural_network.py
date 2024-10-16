#!/usr/bin/env python3
"""This module creates a deep neural network class"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def evaluate(self, X, Y):
        """Function that evaluates the neural network's predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.round(A).astype(int)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Function that calculates one pass of gradient
        descent on the neural network"""
        m = Y.shape[1]
        dZ = cache["A" + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            A = cache["A" + str(i - 1)]
            dW = np.matmul(dZ, A.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.matmul(self.weights["W" + str(i)].T, dZ) * A * (1 - A)
            self.__weights["W" + str(i)] = \
                self.weights["W" + str(i)] - alpha * dW
            self.__weights["b" + str(i)] = \
                self.weights["b" + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Function that trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph is True:
                    costs.append(cost)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object."""
        try:
            with open(filename, 'rb') as f:
                loaded = pickle.load(f)
            return loaded
        except FileNotFoundError:
            return None
