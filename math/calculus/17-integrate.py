#!/usr/bin/env python3
def poly_integral(poly, C=0):
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1 and poly[0] == 0:
        return [0]
    elif len(poly) == 1 and poly[0] != 0:
        return None
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
