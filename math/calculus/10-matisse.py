#!/usr/bin/env python3
"""
poly derivate
"""


def poly_derivative(poly):
    """

    """
    if type(poly) is not list:
        return None
    n = len(poly)
    if n == 0:
        return None
    if n == 1:
        return [0]
    list_item = []
    for i in range(1, n):
        list_item.append(poly[i] * i)
    return list_item
