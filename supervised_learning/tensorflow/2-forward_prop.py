#!/usr/bin/env python3
"""Module for forward_prop function"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""

    a = x

    for i, size in enumerate(layer_sizes):
        # Provide a default activation function if none is specified
        activation = tf.nn.relu if i >= len(activations) else activations[i]

        # Ensure the activation function is valid
        if not callable(activation):
            raise ValueError(f"Invalid activation function at index {i}")

        a = create_layer(
            prev=a,
            n=size,
            activation=activation
        )

    return a
