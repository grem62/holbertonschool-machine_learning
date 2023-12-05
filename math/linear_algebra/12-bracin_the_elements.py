#!/usr/bin/env python3
"""
_summary_
"""


def np_elementwise(mat1, mat2):
    """

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
    """
    addition = mat1 + mat2
    subtraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return addition, subtraction, multiplication, division
