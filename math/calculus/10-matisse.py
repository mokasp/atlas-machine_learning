#!/usr/bin/env python3
def poly_derivative(poly):
    deriv = []
    for i in range(1, len(poly)):
        deriv.append(poly[i] * i)
    if len(deriv) == 0:
        return [0]
    return deriv
