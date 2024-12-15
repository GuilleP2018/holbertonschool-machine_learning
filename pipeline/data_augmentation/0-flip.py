#!/usr/bin/env python3
"""
This module contains the function flip_image
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.
    """
    return tf.image.flip_left_right(image)
