#!/usr/bin/env python3
import numpy as np

def definiteness(matrix):
    if type(matrix) != type(np.array([])):
        raise TypeError('matrix must be a np.ndarray')
    if matrix.shape == (0,) or matrix.shape[0] != matrix.shape[1]:
        return None
    a, b = np.linalg.eig(matrix)
    if a[0] > 0 and a[1] > 0:
        return 'Positive definite'
    elif (a[0] == 0 and a[1] > 0) or (a[0] > 0 and a[1] == 0):
        return 'Positive semi-definite'
    elif (a[0] == 0 and a[1] < 0) or (a[0] < 0 and a[1] == 0):
        return 'Negative semi-definite'
    elif a[0] < 0 and a[1] < 0:
        return 'Negative definite'
    elif (a[0] > 0 and a[1] < 0) or (a[0] < 0 and a[1] > 0):
        return 'Indefinite'