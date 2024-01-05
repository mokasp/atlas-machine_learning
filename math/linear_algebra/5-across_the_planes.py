#!/usr/bin/env python3
""" module containing a function that adds 2d matrices """


def add_matrices2D(mat1, mat2):
    """ func that returns the sum of two 2d matrices """
    if len(mat1) == 0 and len(mat2) == 0:
        return None
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        new_mat = [[0 for _ in range(len(mat1))] for _ in range(len(mat1[0]))]
        for x in range(len(mat1)):
            for y in range(len(mat1[0])):
                new_mat[x][y] = mat1[x][y] + mat2[x][y]
        return new_mat
