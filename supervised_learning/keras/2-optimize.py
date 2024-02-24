#!/usr/bin/env python3
""" module containing function that sets up Adam optimization for a keras
    model with categorical crossentropy loss and accuracy metrics """


def optimize_model(network, alpha, beta1, beta2):
    """ function that sets up Adam optimization for a keras model with
        categorical crossentropy loss and accuracy metrics

        PARAMETERS
        ==========
            network []: the model to optimize
            alpha []: the learning rate
            beta1 []: the first Adam optimization parameter
            beta2 []: the second Adam optimization parameter

        RETURNS
        =======
            None
    """
