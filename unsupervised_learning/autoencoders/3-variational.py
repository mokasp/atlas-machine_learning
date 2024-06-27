#!/usr/bin/env python3
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """ function that contructs a vanilla autoencoder

        Parameters
        ----------
        input_dims : int
            Dimensions of the model input.
        filters : list
            List containing the number of nodes for each hidden layer in
            the encoder.
        latent_dims : int
            Dimensions of the latent space representation.

        Returns
        -------
        encoder : tensorflow.keras.Model
            The encoder model.
        decoder : tensorflow.keras.Model
            The decoder model.
        autoencoder : tensorflow.keras.Model
            The full autoencoder model.
    """

    def sample(params):
        z_mean, z_logsig = params
        eps = K.random_normal(shape=(K.shape(z_mean)[0], latent_dims), mean=0.0, stddev=0.1)
        out = z_mean + K.exp(z_logsig)

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
    points = layers.Lambda(sample)([z_mean, z_logsig])
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
    kl_div = K.sum(1 + z_logsig - K.square(z_mean) - K.exp(z_logsig), axis=-1) * 0.5
    total_loss = K.mean(r_loss + kl_div)



    # compile
    autoencode.compile(optimizer='adam', loss=total_loss)

    return encode, decode, autoencode
