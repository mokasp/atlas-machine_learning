#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    create_layer = __import__('1-create_layer').create_layer

    for lay in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[lay], activations[lay])
    
    return x