#!/usr/bin/env python3
def poly_integral(poly, c=0):
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    integral = [c]
    for i in range(0, len(poly)):
        if i == 0:
            integral.append(poly[i])
        else:
            integral.append(poly[i] / (i + 1))
    if len(integral) == 0:
        return [0]
    return integral
