#!/usr/bin/env python3
import numpy as np
def summation_i_squared(n):
    the_list = np.arange(start=1, stop=n + 1, step=1)
    powered = np.square(the_list)
    return sum(powered)
