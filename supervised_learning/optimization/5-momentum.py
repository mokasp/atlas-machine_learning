#!/usr/bin/env python3
""" module containing function that updates a variable using the gradient
    descent with momentum optimization algorithm """
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using the gradient descent with momentum
        optimization algorithm

        PARAMETERS:
            alpha [float]: the learning rate
            beta1 [float]: the momentum weight
            var [np.ndarray]: the variable to be updated
            grad [np.ndarray]: the gradient of var
            v [np.ndarray]: the previous first moment of var

        RETURNS:
            new_var, new_mom [np.ndarray]: updated variable and the new
                                            momentum

    """
    new_mom = beta1 * v + (1 - beta1) * grad
    new_var = var - alpha * new_mom
    return new_var, new_mom
