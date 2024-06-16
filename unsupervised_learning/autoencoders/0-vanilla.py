#!/usr/bin/env python3
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):

    layers = len(hidden_layers)
    decode_layers = hidden_layers[::-1]

    # encoder
    encode_inpt = K.layers.Input(shape=(input_dims,))
    layer = encode_inpt
    for i in range(layers):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')(layer)
    latent = K.layers.Dense(units=latent_dims, activation='relu')(layer)
    encode = K.Model(inputs=encode_inpt, outputs=latent)

    # decoder
    inpt = K.layers.Input(shape=(latent_dims,))
    layer = inpt
    for i in range(layers):
        layer = K.layers.Dense(units=decode_layers[i], activation='relu')(layer)
    latent = K.layers.Dense(units=input_dims, activation='sigmoid')(layer)
    decode = K.Model(inputs=inpt, outputs=latent)

    # full autoencoder
    autoencode = K.Model(inputs=encode_inpt, outputs=decode(encode(encode_inpt)))

    # compile
    autoencode.compile(optimizer='adam', loss=K.losses.BinaryCrossentropy())

    return encode, decode, autoencode