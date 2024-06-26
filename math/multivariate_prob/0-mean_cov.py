#!/usr/bin/env python3
""" tbw """
import numpy as np


def mean_cov(X):
    """ tbw """
    if (isinstance(X, type(np.array([])))
            and len(X.shape) > 1 and X.shape[0] < 2):
        raise ValueError('X must contain multiple data points')
    if not isinstance(X, type(np.array([]))) or len(X.shape) < 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n = len(X)
    d = len(X[0])

    mean = []

    cov = np.zeros((d, d))
    for i in range(d):
        s = 0
        for j in range(n):
            s += X[j][i]
        mean.append(round(s / n, 8))

    for i in range(d):
        for j in range(i, d):
            s = 0
            for h in range(n):
                s += (X[h][i] - mean[i]) * (X[h][j] - mean[j])
                cov[i][j] = s / (n - 1)
                cov[j][i] = round(cov[i][j], 8)
    return np.array([mean]), np.array(cov)
