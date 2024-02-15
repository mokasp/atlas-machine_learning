#!/usr/bin/env python3
""" module containing function that creates a learning rate decay operation
    in tensorflow using inverse time decay """
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """  creates a learning rate decay operation in tensorflow using
            inverse time decay

        PARAMETERS:
            alpha [float]: the learning rate
            decay_rate [float]: the weight used to determine the
                                rate at which alpha will decay
            gloabal_step [?]: the number of passes of gradient descent
                                that have elapsed
            decay_step [?]: the number of passes of gradient descent that
                            should occur before alpha is decayed further

        RETURNS:
            train [tensor operation]: the learning rate decay operation

    """
    optim = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)
    return optim
