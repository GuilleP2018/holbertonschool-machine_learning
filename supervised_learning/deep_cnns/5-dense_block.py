#!/usr/bin/env python3
"""Module that builds a dense block as described in
Densely Connected Convolutional Networks
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected
    Convolutional Networks
    """
    for i in range(layers):
        X_copy = X
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                            kernel_initializer=K.initializers.he_normal(
                                seed=0))(X)
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                            kernel_initializer=K.initializers.he_normal(
                                seed=0))(X)
        X = K.layers.concatenate([X_copy, X])
        nb_filters += growth_rate

    return X, nb_filters
