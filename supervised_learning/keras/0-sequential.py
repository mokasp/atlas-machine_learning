#!/usr/bin/env python3
""" module containing function that builds a neural network with
    the Keras library """


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ function that builds a neural network with the Keras library

        PARAMETERS
        ==========
            nx [int]: number of input features to the network
            layers [list]: contains the number of nodes in each layer of
                        the network
            activations [list]: contains the activation functions used for
                            each layer of the network
            lambtha [floay]: L2 regularization parameter
            keep_prob [float]: probability that a node will be kept for dropout

        RETURNS
        =======
            the model
    """
