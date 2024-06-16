#!/usr/bin/env python3
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):

    # encoder
    inpt = K.layers.Input(shape=input_dims)
    first_hidden = K.layers.Dense(hidden_layers[0], activation='relu')(inpt)
    second_hidden = K.layers.Dense(hidden_layers[1], activation='relu')(first_hidden)
    latent = K.layers.Dense(latent_dims, activation='relu')(second_hidden)
    encode = K.Model(inputs=inpt, outputs=latent)

    # decoder
    inpt = K.layers.Input(shape=latent_dims)
    first_hidden = K.layers.Dense(hidden_layers[1], activation='relu')(inpt)
    second_hidden = K.layers.Dense(hidden_layers[0], activation='relu')(first_hidden)
    latent = K.layers.Dense(input_dims, activation='sigmoid')(second_hidden)
    decode = K.Model(inputs=inpt, outputs=latent)

    # full autoencoder
    autoencode = K.Model(inputs=encode.input, outputs=decode(encode.output))

    # compile
    autoencode.compile(optimizer='adam', loss=K.losses.BinaryCrossentropy())

    return encode, decode, autoencode