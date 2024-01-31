#!/usr/bin/env python3
import numpy as np

def one_hot_decode(one_hot):
    decoded = []
    try:
        for x in range(one_hot.shape[1]):
            for y in range(len(one_hot.shape[0])):
                if one_hot[y][x] == 1:
                    decoded.append(y)
        return decoded
    except:
        return None