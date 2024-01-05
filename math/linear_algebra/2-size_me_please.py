#!/usr/bin/env python3
""" module containing function that determines the
    shape of a matrix"""


def matrix_shape(matrix):
    """ func that returns shape of a matrix """
    if isinstance(matrix, list):
        shape_list = [len(matrix)]
        if isinstance(matrix[0], list):
            shape_list += matrix_shape(matrix[0])
        else:
            shape_list += []
    return shape_list
