#!/usr/bin/env python3
""" module containing function that """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ function that

        PARAMETERS
        ==========
            []:
            []:

        RETURNS
        =======
            []:
    """
    kern = kernel.shape[0]
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    op = np.zeros((m, h, w))
    pad = np.pad(images[:],  ((0, 0), (1, 1), (1, 1)), mode='constant')
    for row in range(h):
        for col in range(w):
            cur = pad[:, row:row+kern, col:col+kern]
            op[:, row, col] = np.sum(np.multiply(cur, kernel), axis=(1, 2))
    return op
