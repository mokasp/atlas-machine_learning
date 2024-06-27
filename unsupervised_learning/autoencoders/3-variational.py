#!/usr/bin/env python3
import tensorflow.keras as K
import tensorflow as tf


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

    class Encoder(tf.keras.Model):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.hidden_layers = [
                tf.keras.layers.Dense(
                    units=f,
                    activation='relu') for f in filters]
            self.z_mean = tf.keras.layers.Dense(units=latent_dims)
            self.z_logsig = tf.keras.layers.Dense(units=latent_dims)

        def __call__(self, x):
            for layer in self.hidden_layers:
                x = layer(x)
            z_mean = self.z_mean(x)
            z_logsig = self.z_logsig(x)
            return z_mean, z_logsig

    class Decoder(tf.keras.Model):
        def __init__(self, name=None):
            super().__init__(name=name)
            decode_layers = filters[::-1]
            self.hidden_layers = [
                tf.keras.layers.Dense(
                    units=f,
                    activation='relu') for f in decode_layers]
            self.output_layer = tf.keras.layers.Dense(
                units=input_dims, activation='sigmoid')

        def __call__(self, x):
            for layer in self.hidden_layers:
                x = layer(x)
            return self.output_layer(x)

    class VariationalAutoencoder(tf.keras.Model):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.encoder = Encoder()
            self.decoder = Decoder()

        def reparameterize(self, z_mean, z_logsig):
            eps = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_logsig) * eps

        def __call__(self, x):
            z_mean, z_logsig = self.encoder(x)
            z = self.reparameterize(z_mean, z_logsig)
            x_recon = self.decoder(z)
            return x_recon, z_mean, z_logsig

    def compute_loss(x, x_recon, z_mean, z_logsig):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(
                    x, x_recon), axis=1))
        kl_divergence = -0.5 * \
            tf.reduce_mean(
                tf.reduce_sum(
                    1 +
                    z_logsig -
                    tf.square(z_mean) -
                    tf.exp(z_logsig),
                    axis=1))
        return reconstruction_loss + kl_divergence

    encode = Encoder()
    decode = Decoder()
    autoencode = VariationalAutoencoder()

    autoencode.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=lambda x,
        x_recon: compute_loss(
            x,
            x_recon,
            encode.z_mean,
            encode.z_logsig))

    return encode, decode, autoencode
