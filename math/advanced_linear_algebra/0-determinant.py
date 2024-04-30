#!/usr/bin/env python3
import numpy as np

# [[a b c]
#  [e f g]
#  [h i j]]

# (a * ((f * j) - (g * i))) - (b * ((e * j) - (g * h))) + (c * ((e * i) - (f * h)))

def determinant(matrix):
    if type(matrix) is not list or len(matrix) == 0 or type(matrix[0]) is not list:
        raise TypeError('matrix must be a list of lists')
    if len(matrix[0]) == 0:
        return 1
    if len(matrix) == len(matrix[0]):
        return round(np.linalg.det(np.array(matrix)))
    else:
        raise ValueError('matrix must be a square matrix')