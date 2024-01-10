#!/usr/bin/env python3
""" module containing function that finds
    the derivative of a polynomials """


def poly_derivative(poly):
    """ func that returns a list of coefficents
        of the derivative of a polynomial """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    deriv = []
    for i in range(1, len(poly)):
        deriv.append(poly[i] * i)
    if len(deriv) == 0:
        return [0]
    return deriv
