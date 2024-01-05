#!/usr/bin/env python3
""" module containg function that does matrix multiplication """


def mat_mul(mat1, mat2):
    """ func that returns the product of two 2d matricies"""
    mat1_rows = len(mat1)
    mat1_columns = len(mat1[0])
    mat2_rows = len(mat2)
    mat2_columns = len(mat2[0])
    mat3 = []
    if mat1_columns != mat2_rows:
        return None
    for x in range(mat1_rows):
        mat3_temp = []
        for y in range(mat2_columns):
            mat3_temp.append(0)
        mat3.append(mat3_temp)

    for x in range(mat1_rows):
        for y in range(mat2_columns):
            for z in range(mat1_columns):
                mat3[x][y] += mat1[x][z] * mat2[z][y]
    return mat3
