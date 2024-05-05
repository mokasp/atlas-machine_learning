#!/usr/bin/env python3
""" tbw """
import numpy as np


class MultiNormal():
    """ tbw """

    def __init__(self, data):
        """ tbw """
        if not isinstance(data, type(np.array([]))) or len(data.shape) < 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if isinstance(data, type(np.array([]))) and data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        n = data.shape[1]
        d = data.shape[0]

        dat = data.T

        self.mean = np.mean(data, axis=1, keepdims=True)

        cov = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                s = 0
                for h in range(n):
                    s += (dat[h][i] - self.mean[i]) * \
                        (dat[h][j] - self.mean[j])
                    cov[i][j] = s / (n - 1)
                    cov[j][i] = round(cov[i][j], 8)

        self.cov = cov

    def pdf(self, x):
        """ tbw """

        if not isinstance(x, type(np.array([]))):
            raise TypeError('x must be a numpy.ndarray')

        d = len(self.mean)

        if x.shape[0] != d or x.shape[1] != 1:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        coeff = 1.0 / ((2 * np.pi) ** (d / 2) *
                       np.sqrt(np.linalg.det(self.cov)))

        expo = (-1 / 2) * \
            ((x - self.mean).T.dot(np.linalg.inv(self.cov).dot(x - self.mean)))

        return float(coeff * np.exp(expo))
