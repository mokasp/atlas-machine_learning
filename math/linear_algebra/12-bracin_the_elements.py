#!/usr/bin/env python3
""" module containing a function that does elementwise
    operations on two 2d matricies"""


def np_elementwise(mat1, mat2):
    """ func that uses numpy to perform elementwise operations
        on two 2d matricies"""
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
