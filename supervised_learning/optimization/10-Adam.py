#!/usr/bin/env python3
""" module containing function that creates the training operation for a
    neural network in tensorflow using the Adam optimization algorithm """
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """  creates the training operation for a neural network in tensorflow
            using the Adam optimization algorithm

        PARAMETERS:
            loss [float]: loss of the network
            alpha [float]: the learning rate
            beta1 [float]: the weight used for the first moment
            beta2 [float]: the weight used for the second moment
            epsilon [float]: small number to avoid division by zero

        RETURNS:
            train [tensor operation]: the Adam optimization algorithm

    """
    optim = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                   beta2=beta2, epsilon=epsilon)
    train = optim.minimize(loss)
    return train
