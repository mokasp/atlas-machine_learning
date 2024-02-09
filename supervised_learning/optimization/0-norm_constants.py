#!/usr/bin/env python3
""" module containing function that calculates the normalization
    constants of a matrix """
import numpy as np


def normalization_constants(X):
    """ calculates the normalization (standardization) constants of a matrix

         PARAMETERS:
            X [np.ndarray] - np array of shape (m, nx) to be normalized
                            (m - number of data points)
                            (nx - number of features)

    """
    return np.mean(X, axis=0), np.std(X, axis=0)
