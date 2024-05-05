#!/usr/bin/env python3
"""Module for calculate_accuracy function"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction"""
    y_pred_cls = tf.argmax(y_pred, 1)
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
