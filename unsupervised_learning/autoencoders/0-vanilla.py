#!usr/bin/env python3
import numpy as np
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):

    # encoder
    inpt = tf.keras.layers.Input(shape=input_dims)
    first_hidden = tf.keras.layers.Dense(hidden_layers[0], activation='relu')(inpt)
    second_hidden = tf.keras.layers.Dense(hidden_layers[1], activation='relu')(first_hidden)
    latent = tf.keras.layers.Dense(latent_dims, activation='relu')(second_hidden)
    encode = tf.keras.Model(inputs=inpt, outputs=latent)

    # decoder
    inpt = tf.keras.layers.Input(shape=latent_dims)
    first_hidden = tf.keras.layers.Dense(hidden_layers[1], activation='relu')(inpt)
    second_hidden = tf.keras.layers.Dense(hidden_layers[0], activation='relu')(first_hidden)
    latent = tf.keras.layers.Dense(input_dims, activation='sigmoid')(second_hidden)
    decode = tf.keras.Model(inputs=inpt, outputs=latent)

    # full autoencoder
    autoencode = tf.keras.Model(inputs=encode.input, outputs=decode(encode.output))

    # compile
    autoencode.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())

    return encode, decode, autoencode