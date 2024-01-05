#!/usr/bin/env python3
""" module containing a function that concatenates two 2d matricies """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ func that uses numpy to return the concatenation of two
        2d matricies """
    return np.concatenate((mat1, mat2), axis=axis)
