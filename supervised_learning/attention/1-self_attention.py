#!/usr/bin/env python3
"""
Self Attention
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    This class represents the self-attention mechanism for machine translation
    """

    def __init__(self, units):
        """
        Initializes the SelfAttention layer.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Forward pass to compute attention mechanism.
        """
        # Expand dimensions of the previous hidden state for broadcasting
        s_prev_expanded = tf.expand_dims(input=s_prev, axis=1)

        # Comput scalar score for each time step, softmaxed to get weights
        scores = self.V(
            tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        weights = tf.nn.softmax(scores, axis=1)

        # Weighted sum of encoder hidden states (reduce over input_seq_len dim)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
