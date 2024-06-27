#!/usr/bin/env python3
""" tbw """
import tensorflow.keras as K
import tensorflow as tf


def autoencoder(input_dims, filters, latent_dims):
    """ tbw """

    def sample(params):
        """ tbw """
        z_mean, z_logsig = params
        eps = K.backend.random_normal(
            shape=(
                K.backend.shape(z_mean)[0],
                K.backend.int_shape(z_mean)[1]),
            mean=0.0,
            stddev=0.1)
        out = z_mean + K.backend.exp(z_logsig)

        return out * eps

    layers = len(filters)
    decode_layers = filters[::-1]

    # encoder
    encode_inpt = K.layers.Input(shape=(input_dims,))
    layer = encode_inpt
    for i in range(layers):
        layer = K.layers.Dense(units=filters[i],
                               activation='relu')(layer)
    z_mean = K.layers.Dense(units=latent_dims)(layer)
    z_logsig = K.layers.Dense(units=latent_dims)(layer)
    points = K.layers.Lambda(sample, output_shape=(
        latent_dims,))([z_mean, z_logsig])
    encode = K.Model(inputs=encode_inpt, outputs=[z_mean, z_logsig, points])

    # decoder
    inpt = K.layers.Input(shape=(latent_dims,))
    layer = inpt
    for i in range(layers):
        layer = K.layers.Dense(units=decode_layers[i],
                               activation='relu')(layer)
    latent = K.layers.Dense(units=input_dims, activation='sigmoid')(layer)
    decode = K.Model(inputs=inpt, outputs=latent)

    # full autoencoder
    autoencode = K.Model(inputs=encode_inpt, outputs=decode(encode
                                                            (encode_inpt)[2]))

    r_loss = K.losses.binary_crossentropy(encode_inpt, latent) * input_dims
    kl_div = K.backend.sum(
        1 + z_logsig - K.backend.square(z_mean) - K.backend.exp(z_logsig),
        axis=-1) * -0.5
    total_loss = K.backend.mean(r_loss + kl_div)

    # compile
    autoencode.compile(optimizer='adam', loss=total_loss)

    return encode, decode, autoencode
