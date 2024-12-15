#!/usr/bin/env python3
"""
This module contains the function change_brightness
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.
    """
    return tf.image.random_brightness(image, max_delta)
