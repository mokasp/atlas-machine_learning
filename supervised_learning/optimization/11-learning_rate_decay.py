#!/usr/bin/env python3
""" module containing function that updates the learning
    rate using inverse time decay in numpy """
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates the learning rate using inverse time decay in numpy

        PARAMETERS:
            alpha [float]: the learning rate
            decay_rate [float]: the weight used to determine the
                                rate at which alpha will decay
            gloabal_step [?]: the number of passes of gradient descent
                                that have elapsed
            decay_step [?]: the number of passes of gradient descent that
                            should occur before alpha is decayed further

        RETURNS:
            alpha [float]: updated value for alpha

    """
    if global_step < decay_step:
        return alpha
    else:
        step = int(global_step / decay_step)
        alpha = alpha * (1 / (1 + decay_rate * step))
    return alpha
