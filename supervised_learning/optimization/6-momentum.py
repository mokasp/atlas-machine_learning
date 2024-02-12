#!/usr/bin/env python3
""" module containing function that creates the training operation for a
    neural network in tensorflow using the gradient descent with momentum
    optimization algorithm """
import numpy as np
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """ creates the training operation for a neural network in tensorflow
        using the gradient descent with momentum optimization algorithm

        PARAMETERS:
            loss [float]: loss of the network
            alpha [float]: the learning rate
            beta1 [float]: the momentum weight

        RETURNS:
            train [tensor operation]: momentum optimization operation

    """
    optim = tf.train.MomentumOptimizer(alpha, beta1)
    train = optim.minimize(loss)
    return train
