#!/usr/bin/env python3
def summation_i_squared(n):
    if not isinstance(n, int) or n <= 0:
        return None
    the_list = list(range(n + 1))
    powered = map(lambda x: x ** 2, the_list)
    return sum(powered)
