#!/usr/bin/env python3
"""
calculate integral polynomial
"""


def poly_integral(poly, c=0):
    if type(poly) is not list:
        return None
    if type(c) is not int:
        return None
    n = len(poly)
    if n == 0:
        return None

    integral = [c]

    for i in range(n):
        integral.append(poly[i] / (i + 1))

    return integral
