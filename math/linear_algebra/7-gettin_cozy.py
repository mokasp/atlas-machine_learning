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
    new_mat2 = []
    new_mat3 = []
    if axis == 0:
        for x in range(mat1_x):
            for y in range(mat1_y):
                if x < mat1_x / 2:
                    new_mat1.append(mat01[x][y])
                else:
                    new_mat2.append(mat01[x][y])
        for x in range(mat2_x):
            for y in range(mat2_y):
                if x < mat2_x / 2:
                    new_mat3.append(mat02[x][y])
                else:
                    new_mat3.append(mat02[x][y])
        new_mat.append(new_mat1)
        new_mat.append(new_mat2)
        new_mat.append(new_mat3)
        return new_mat
    else:
        return None