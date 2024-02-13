#!/usr/bin/env python3
""" module containing function that normalizes an unactivated output of a
    neural network using batch normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output of a neural network
        using batch normalization

        PARAMETERS:
            z [np.ndarray]: array of shape (m, n) that should be normalized
                            m [int]: number of data points
                            n [int]: number of features in Z
            gamma [np.ndarray]: array of shape (1, n) containing the
                                scales used for batch normalization
            beta [np.ndarry]: array of shape (1, n) containing the
                                offsets used for batch normalization
            epsilon [?]: a small number used to avoid division by zero

        RETURNS:
            alpha [float]: updated value for alpha

    """
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    z_norm = (Z - mean) / np.sqrt(var + epsilon)
    out = gamma * z_norm + beta
    return out
