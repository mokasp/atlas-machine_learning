#!/usr/bin/env python3
import numpy as np

def definiteness(matrix):
    if type(matrix) != type(np.array([])):
        raise TypeError('matrix must be a numpy.ndarray')
    elif matrix.shape == (0,) or matrix.shape[0] != matrix.shape[1]:
        return None
    elif matrix[0][1] != matrix[1][0]:
        return None
    values, _ = np.linalg.eig(matrix)
    if values[0] > 0 and values[1] > 0:
        return 'Positive definite'
    elif (values[0] == 0 and values[1] > 0) or (values[0] > 0 and values[1] == 0):
        return 'Positive semi-definite'
    elif (values[0] == 0 and values[1] < 0) or (values[0] < 0 and values[1] == 0):
        return 'Negative semi-definite'
    elif values[0] < 0 and values[1] < 0:
        return 'Negative definite'
    elif (values[0] > 0 and values[1] < 0) or (values[0] < 0 and values[1] > 0):
        return 'Indefinite'