#!/usr/bin/env python3
import numpy as np


def pdf(X,m, S):
    k = len(m)

    mm = m.reshape(k, 1)

    # constant
    const = (2 * np.pi) ** k

    # determinate of covariance mat
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    # scaling
    scale = 1.0 / np.sqrt(const * det)

    diff = (X - mm.T)

    exp_term = -0.5 * (np.dot(diff, inv).dot(diff.T))

    res = scale * np.exp(exp_term)

    row, col = res.shape
    diag = np.eye(row, col, dtype=bool)

    return res[diag]