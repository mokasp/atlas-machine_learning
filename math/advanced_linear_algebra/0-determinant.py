#!/usr/bin/env python3

# [[a b c]
#  [e f g]
#  [h i j]]

# (a * ((f * j) - (g * i))) - (b * ((e * j) - (g * h))) + (c * ((e * i) - (f * h)))

def minor(matrix, col):
    temp = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != 0 and j != col:
                temp.append(matrix[i][j])
    return temp

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
            temp2 = 0
            for i in range(len(matrix)):
                mino = minor(matrix, i)
                a = matrix[0][i] * ((mino[0] * mino[3]) - (mino[1] * mino[2]))
                if i != 1:
                    temp2 += a
                else:
                    temp2 -= a
            return temp2
    else:
        raise ValueError('matrix must be a square matrix')