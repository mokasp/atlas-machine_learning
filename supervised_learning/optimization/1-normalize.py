#!/usr/bin/env python3
""" module containing function that normalizes a matrix """
import numpy as np


def normalize(X, m, s):
    """ calculates the normalizes standardizes) a matrix

         PARAMETERS:
            X [np.ndarray]: np array of shape (m, nx) to be normalized
                            m - number of data points
                            nx - number of features
            m [np.ndarray]: np array of shape (nx, ) that contains the
                            mean of all features X
            m [np.ndarray]: np array of shape (nx, ) that contains the
                            standard deviation of all features X

    """
    return (X - m) / s
