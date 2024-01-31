#!/usr/bin/env python3
import numpy as np

def one_hot_decode(one_hot):
    decoded = []
    for x in range(len(one_hot[0])):
        for y in range(len(one_hot)):
            if one_hot[y][x] == 1:
                decoded.append(y)
    return decoded