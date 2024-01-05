#!/usr/bin/env python3
""" module containing function that transposes a matrix """


def matrix_transpose(matrix):
    """ func that transposes a 2d matrix"""
    return [list(i) for i in zip(*matrix)]
