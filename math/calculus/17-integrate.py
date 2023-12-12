#!/usr/bin/env python3
"""
calculate integral polynomial
"""

def poly_integral(poly, C=0):
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly) or not isinstance(C, int):
        return None

    integral = [C]

    for i, coeff in enumerate(poly):
        power = i + 1
        term = coeff / power

        if term.is_integer():
            integral.append(int(term))
        else:
            integral.append(term)

    # Remove trailing zeros
    while integral and integral[-1] == 0:
        integral.pop()

    return integral
