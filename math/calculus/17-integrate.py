#!/usr/bin/env python3
""" module containing function that finds
    the integral of a polynomial"""


def poly_integral(poly, C=0):
    """ func that returns a list of coefficents
        of the integral of a polynomial """
    if not isinstance(poly, list) or len(poly) == 0 or C is None:
        return None
    if len(poly) == 1 and poly[0] == 0 and C != 0:
        return [C]
    elif len(poly) == 1 and poly[0] == 0:
        return [0]
    integral = [C]
    for i in range(0, len(poly)):
        if i == 0:
            integral.append(poly[i])
        else:
            elem = poly[i] / (i + 1)
            if elem.is_integer():
                elem = int(elem)
            integral.append(elem)
    if len(integral) == 0:
        return [0]
    return integral
