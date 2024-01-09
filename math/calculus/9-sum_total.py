#!/usr/bin/env python3
def summation_i_squared(n):
    the_list = list(range(n + 1))
    powered = map(lambda x: x ** 2, the_list)
    return sum(powered)
