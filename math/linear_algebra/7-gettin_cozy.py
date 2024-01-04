#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    mat01 = mat1
    mat02 = mat2
    mat1_x = len(mat01)
    mat1_y = len(mat01[0])
    mat2_x = len(mat02)
    mat2_y = len(mat02[0])
    new_mat = []
    new_mat1 = []
    if axis == 0:
        for x in range(mat1_x):
            new_mat1 = []
            for y in range(mat1_y):
                new_mat1.append(mat01[x][y])
            if len(new_mat1) != 0:
                new_mat.append(new_mat1)
        for x in range(mat2_x):
            new_mat4 = []
            for y in range(mat2_y):
                new_mat4.append(mat02[x][y])
            if len(new_mat4) != 0:
                new_mat.append(new_mat4)
        return new_mat
    else:
        return None