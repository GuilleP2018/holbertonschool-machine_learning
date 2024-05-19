#!/usr/bin/env python3
"""This module contains the function for updating a variable using the RMSProp
optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Set up the RMSProp optimization algorithm in TensorFlow.
    """
    # Create a RMSprop optimizer object with the specified
    # learning rate, RMSProp weight, and epsilon
    optimizer = \
        tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2,
                                    epsilon=epsilon)

    # Return the optimizer
    return optimizer
