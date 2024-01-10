#!/usr/bin/env python3
""" module containing function that finds the
    summation of a series"""


def summation_i_squared(n):
    """ func that returns the sum of all elements in a series """
    if not isinstance(n, int) or n <= 0:
        return None
    the_list = list(range(n + 1))
    powered = map(lambda x: x ** 2, the_list)
    return sum(powered)
