#!/usr/bin/env python3
"""
sigma sumation
"""


def summation_i_squared(n):
    """
    operation squared in range 1-5
    """
    if (type(n) is not int) or (n < 1):
        return None
    else:
        somme = n * (n + 1) * (2 * n + 1) // 6
    return somme
