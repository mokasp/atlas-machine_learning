#!/usr/bin/env python3
"""" module containing function that builds the DenseNet-121 architecture """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ builds the DenseNet-121 architecture

    Parameters:
        growth_rate [int]: Growth rate.
        compression [float]: Compression factor.

    Returns:
        keras.Model: The constructed DenseNet-121 model.
    """
    inp = K.layers.Input((224, 224, 3))
    init = K.initializers.HeNormal()
    filters = 64
    x = K.layers.BatchNormalization(axis=3)(inp)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters, (7, 7), strides=(2, 2),
                         kernel_initializer=init,
                         padding='same')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                               padding='same')(x)
    x, filters = dense_block(x, filters, growth_rate, 6)
    x, filters = transition_layer(x, filters, compression)
    x, filters = dense_block(x, filters, growth_rate, 12)
    x, filters = transition_layer(x, filters, compression)
    x, filters = dense_block(x, filters, growth_rate, 24)
    x, filters = transition_layer(x, filters, compression)
    x, filters = dense_block(x, filters, growth_rate, 16)
    x = K.layers.AveragePooling2D((7, 7), padding='valid', strides=(1, 1))(x)
    x = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(x)
    return K.Model(inp, x)
