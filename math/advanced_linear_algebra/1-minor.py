#!/usr/bin/env python3
""" module containing function that calculates the minor matrix of a given
    square matrix """


def minor(matrix):
    """
    function that calculates the minor matrix of a given square matrix.

    Args:
        matrix (list of lists): A square matrix whose minor matrix is to be
        calculated.

    Returns:
        list of lists: The minor matrix of the input matrix.
    """

    # first store length of outer list
    length = len(matrix)

    # if length is 0, raise value error indicating the list does not contain
    # other lists
    if length == 0:
        raise TypeError('matrix must be a list of lists')

    # if length is greater than 0, check all types and lengths of inner items
    else:
        for i in range(length):
            if len(matrix[i]) != length:
                raise ValueError('matrix must be a square matrix')
            elif not isinstance(matrix[i], list):
                raise TypeError('matrix must be a list of lists')

            # if both inner and outer list have length of 1, return a list
            # containing 1 in a list
            elif len(matrix[i]) == 1 and length == 1:
                return [[1]]

    # make copy of matrix
    mnr = [x[:] for x in matrix]

    # if both lengths are 2, switch positions of values to find minor of 2x2
    # matrix
    if length == 2:
        for i in range(length):
            for j in range(length):
                for x in range(length):
                    for y in range(length):
                        if x != i and y != j:
                            mnr[i][j] = matrix[x][y]
        return mnr

    # if lenghts ae greater than two, calculate determinate of each submatrix
    # for each value.
    else:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                b = []
                for x in range(len(matrix)):
                    a = []
                    for y in range(len(matrix[0])):
                        if x != i and y != j:
                            a.append(matrix[x][y])
                    if x != i:
                        b.append(a)
                mnr[i][j] = ((b[0][0] * b[1][1]) - (b[0][1] * b[1][0]))
        return mnr
