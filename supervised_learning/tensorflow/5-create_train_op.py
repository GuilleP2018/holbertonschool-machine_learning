#!/usr/bin/env python3
"""Module for create train op function"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_train_op(loss, alpha):
    """Function that creates the training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train = optimizer.minimize(loss)
    return train
