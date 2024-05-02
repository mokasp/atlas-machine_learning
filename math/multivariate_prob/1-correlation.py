#!/usr/bin/env python3
import numpy as np


def correlation(C):

    #extract square roots of values along the diagnal
    sqr_diag = np.sqrt(np.diag(C))

    out = np.outer(sqr_diag, sqr_diag)


    return C / out