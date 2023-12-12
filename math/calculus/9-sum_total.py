#!/usr/bin/env python3
"""
sigma sumation
"""


def summation_i_squared(n):
    """
    operation squared in range 1-5
    """
    squared = [i**2 for i in range(1, n + 1)]
    somme = sum(squared)
    return somme
