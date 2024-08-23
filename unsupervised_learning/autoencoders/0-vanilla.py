#!/usr/bin/env python3
"""This modlue contains a vanilla autoencoder function"""
from tensorflow import keras as keras


def autoencoder(input_dim, hidden_layers, laten_dims):
    """
    This function creates a vanilla autoencoder
    """
    input_encoder = keras.layers.Input(shape=(input_dim,))
    input_encoded = input_encoder
    for layer in hidden_layers:
        input_encoded = keras.layers.Dense(
            layer, activation='relu')(input_encoded)
    latent = keras.layers.Dense(laten_dims, activation='relu')(input_encoded)
    encoder = keras.models.Model(input_encoder, latent)

    input_decoder = keras.layers.Input(shape=(laten_dims,))
    input_decoded = input_decoder
    for layer in hidden_layers[::-1]:
        input_decoded = keras.layers.Dense(
            layer, activation='relu')(input_decoded)
    decoded = keras.layers.Dense(
        input_dim, activation='sigmoid')(input_decoded)
    decoder = keras.models.Model(input_decoder, decoded)

    input_auto = keras.layers.Input(shape=(input_dim,))
    encoder_out = encoder(input_auto)
    decoder_out = decoder(encoder_out)
    auto = keras.models.Model(input_auto, decoder_out)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
