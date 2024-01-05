#!/usr/bin/env python3
""" module containing a function that adds 2d matrices """


def add_matrices2D(mat1, mat2):
    """ func that returns the sum of two 2d matrices """
    if len(mat1) == 0 or len(mat2) == 0:
        return [[], []]
    elif len(mat1[0]) == 0 or len(mat2[0]) == 0:
        return [[], []]
    if get_shape(mat1) == get_shape(mat2):
        return [[mat1[x][y] + mat2[x][y]
                 for y in range(len(mat1[0]))] for x in range(len(mat1))]


def get_shape(matrix):
    """ gets shape of a matrix """
    if isinstance(matrix, list):
        shape_list = [len(matrix)]
        if isinstance(matrix[0], list):
            shape_list += get_shape(matrix[0])
        else:
            shape_list += []
    return shape_list
