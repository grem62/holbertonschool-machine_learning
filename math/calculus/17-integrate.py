#!/usr/bin/env python3
"""
calculate integral polynomial
"""


def poly_integral(poly, C=0):
    """
    arguments: poly, C
    """
    if type(poly) is not list:
        return None
    if poly == []:
        return None
    if type(C) is not int:
        return None
    n = len(poly)
    if n == 0:
        return None

    integral = [C]

    for i in range(n):
        calcul = poly[i] / (i + 1)
        if isinstance(calcul, float):
            if calcul.is_integer():
                integral.append(int(calcul))
            else:
                integral.append(round(calcul, 16))
        else:
            integral.append(int(calcul))

    return integral
