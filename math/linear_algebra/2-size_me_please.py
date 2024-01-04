#!/usr/bin/env python3
def matrix_shape(matrix):
    if isinstance(matrix, list):
        shape_list = [len(matrix)]
        if isinstance(matrix[0], list):
            shape_list += matrix_shape(matrix[0])
        else:
            shape_list += []
    return shape_list