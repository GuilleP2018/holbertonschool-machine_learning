#!/usr/bin/env python3
"""
RNN encoder
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    A class representing a RNN encoder, inheriting from the keras Layer class
    tensorflow.keras.layers.Layer
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor of RNNEncoder
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Takes the input tensor `x` (of shape `(batch, input_seq_len)`
        containing word indices) and the initial hidden state `initial`
        (of shape `(batch, units)`), feeding the embeddings to the GRU.
        """
        # Convert word indices into embeddings
        x = self.embedding(x)

        # Pass the embeddings and the initial hidden state to the GRU
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
