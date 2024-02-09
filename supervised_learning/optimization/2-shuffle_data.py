
#!/usr/bin/env python3
""" module containing function that shuffles the data points in two matrices
    the same way """
import numpy as np


def shuffle_data(X, Y):
    """ shuffles the datapoints in two maticies the same way

         PARAMETERS:
            X [np.ndarray]: first np array of shape (m, nx) to be shuffled
                            m - number of data points
                            nx - number of features in X
            X [np.ndarray]: second np array of shape (m, ny) to be shuffled
                            m - same number of data points as in X
                            ny - number of features in Y

    """
    return np.random.permutation(X), np.random.permutation(Y)