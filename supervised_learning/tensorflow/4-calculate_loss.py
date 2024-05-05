#!/usr/bin/env python3
"""Module for calculate_loss function"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_loss(y, y_pred):
    """Function that calculates the loss of a prediction"""
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (logits=y_pred, labels=y))
    return loss
