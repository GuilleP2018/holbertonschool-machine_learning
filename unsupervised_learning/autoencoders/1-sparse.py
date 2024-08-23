#!/usr/bin/env python3
"""This modlue contains a sparse autoencoder function"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """This function creates a sparse autoencoder"""
    encoder_inputs = keras.Input(shape=(input_dims,))

    for idx, units in enumerate(hidden_layers):
        layer = keras.layers.Dense(units=units, activation="relu")
        if idx == 0:
            outputs = layer(encoder_inputs)
        else:
            outputs = layer(outputs)
    latent = keras.layers.Dense(
        units=latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha),
    )
    latent = latent(outputs)
    encoder = keras.models.Model(inputs=encoder_inputs, outputs=latent)

    decoder_inputs = keras.Input(shape=(latent_dims,))
    for idx, units in enumerate(reversed(hidden_layers)):
        layer = keras.layers.Dense(units=units, activation="relu")
        if idx == 0:
            outputs = layer(decoder_inputs)
        else:
            outputs = layer(outputs)
    layer = keras.layers.Dense(units=input_dims, activation="sigmoid")
    outputs = layer(outputs)

    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)
    outputs = encoder(encoder_inputs)
    decoded = decoder(outputs)
    auto = keras.models.Model(inputs=encoder_inputs, outputs=decoded)
    auto.compile(optimizer="Adam", loss="binary_crossentropy")

    return encoder, decoder, auto
