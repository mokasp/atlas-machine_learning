#!/usr/bin/env python3
import numpy as np

# def check(P):

def absorbing(P):
    n = len(P)
    if list(np.diag(P)).count(1.0) > 0:
        if P[0][0] == 1:
            if P[1:, 0].any() > 0:
                if P[2:, 1].any() > 0:
                    return True
                else:
                    return False
            if any(val > 0 for val in list(P[:, 0])):
                if n == 2:
                    return True
                else:
                    if absorbing(P[1:, 1:]):
                        return True
    return False