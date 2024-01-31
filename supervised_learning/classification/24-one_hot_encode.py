#!/usr/bin/env python3
""" module containing function that decodes a one hot encoded vector """
import numpy as np


def one_hot_encode(Y, classes):
    """ function that decodes a one hot encoded vector """
    encoded = np.zeros((classes, len(Y)))
    for x, y in enumerate(Y):
        encoded[y][x] = 1
    return encoded
