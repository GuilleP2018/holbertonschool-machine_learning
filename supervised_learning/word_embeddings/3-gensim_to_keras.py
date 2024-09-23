#!/usr/bin/env python3
"""This module has the method gensim_to_keras"""
import tensorflow as tf


def gensim_to_keras(model):
    """This method converts a gensim word2vec model to a
    keras Embedding layer
    """
    weights = model.wv.vectors
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
    return embedding_layer
