#!/usr/bin/env python3
""" module containing function that  creates the training operation
    for a neural network in tensorflow using the RMSProp optimization
    algorithm """
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """  creates the training operation for a neural network in
            tensorflow using the RMSProp optimization algorithm

        PARAMETERS:
            loss [float]: loss of the network
            alpha [float]: the learning rate
            beta2 [float]: the RMSProp weight
            epsilon [?]: small number to avoid division by zero

        RETURNS:
            train [tensor operation]: RMSProp optimization operation

    """
    optim = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                      epsilon=epsilon)
    train = optim.minimize(loss)
    return train
