#!/usr/bin/env python3
import numpy as np

def one_hot_encode(Y, classes):
    encoded = np.zeros((classes, len(Y)))
    for x, y in enumerate(Y):
        encoded[y][x] = 1
    return encoded