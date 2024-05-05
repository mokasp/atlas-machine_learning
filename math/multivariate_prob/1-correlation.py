#!/usr/bin/env python3
import numpy as np


def correlation(C):

    if type(C) != type(np.array([])):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) < 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    #extract square roots of values along the diagnal
    sqr_diag = np.sqrt(np.diag(C))

    out = np.outer(sqr_diag, sqr_diag)


    return C / out