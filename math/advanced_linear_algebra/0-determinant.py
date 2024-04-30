#!/usr/bin/env python3

# [[a b c]
#  [e f g]
#  [h i j]]

# (a * ((f * j) - (g * i))) - (b * ((e * j) - (g * h))) + (c * ((e * i) - (f * h)))

def minor(matrix, col):
    row = []
    mino = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != 0 and j != col:
                row.append(matrix[i][j])
        if i != 0:
            mino.append(row)
            row = []
    return mino

def determinant(matrix):
    if type(matrix) is not list or len(matrix) == 0 or type(matrix[0]) is not list:
        raise TypeError('matrix must be a list of lists')
    if len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == len(matrix[0]):
        if len(matrix) == 2:
            return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
        else:
            det = 0
            for i in range(len(matrix)):
                mino = minor(matrix, i)
                det = det + ((-1) ** i) * matrix[0][i] * determinant(mino)
            return det
    else:
        raise ValueError('matrix must be a square matrix')