#!/usr/bin/env python3
"""
Scaled dot product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.
    """
    # Get dk from last dimension of Q (or K)
    # NOTE cast it to float32 to avoid type issues with tf
    dk = tf.cast(Q.shape[-1], dtype=tf.float32)

    # Matrix multiplication of Q by transposed K
    scores = tf.matmul(Q, K, transpose_b=True)

    # Scale attention scores by square root of dk
    scaled_scores = scores / tf.sqrt(dk)

    # If given a mask, apply it to the scores
    if mask:
        # Masked positions basically get set to a large "-inf" negative value
        scaled_scores += (mask * -1e-9)

    # Softmax gets us the attention weights
    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

    # Multiply the weights by the values V to get the output
    output = tf.matmul(attention_weights, V)

    return output, attention_weights
