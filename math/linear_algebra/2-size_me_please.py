#!/usr/bin/env python3
"""
shape of matrix
"""


def matrix_shape(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
