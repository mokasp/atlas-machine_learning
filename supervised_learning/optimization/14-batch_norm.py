#!/usr/bin/env python3
""" module containing function that creates a batch normalization layer for
    a neural network in tensorflow """
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """  creates a batch normalization layer for a neural network in
            tensorflow

        PARAMETERS:
            prev [?]: activated output of the previous layer
            n [int]: number of nodes in the layer to be created
            activation [?]: activation function that should be used
                            on the output of the layer

        RETURNS:
            train [tensor]: tensor of the activated output for the layer

    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        name="layer",
        kernel_initializer=init
    )
    x = layer(prev)
    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    norm = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)
    activated = tf.keras.layers.Activation(activation)
    return activated(norm)
