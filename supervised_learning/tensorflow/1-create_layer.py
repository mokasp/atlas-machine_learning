#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )
    return layer(prev)