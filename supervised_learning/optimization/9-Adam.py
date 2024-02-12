#!/usr/bin/env python3
""" module containing function that updates a variable in place
    using the Adam optimization algorithm """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ updates a variable in place using the Adam optimization algorithm

        PARAMETERS:
            alpha [float]: the learning rate
            beta1 [float]: the weight used for the first moment
            beta2 [float]: the weight used for the second moment
            epsilon [float]: small number to avoid division by zero
            var [np.ndarray]: the variable to be updated
            grad [np.ndarray]: the gradient of var
            v [np.ndarray]: the previous first moment of var
            s [np.ndarray]: the previous second moment of var
            t [?]: the time step used for bias correction

        RETURNS:
            new_var, new_mom1, new_mom2 [np.ndarray]: updated variable and
            the new first and second moment

    """
    new_mom1 = beta1 * v + (1 - beta1) * grad
    new_mom2 = beta2 * s + (1 - beta2) * (grad ** 2)
    m1 = new_mom1 / (1 - np.power(beta1, t))
    m2 = new_mom2 / (1 - np.power(beta2, t))
    new_var = var - alpha * m1 / (np.sqrt(m2) + epsilon)
    return new_var, new_mom1, new_mom2
