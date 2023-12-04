#!/usr/bin/env python3
"""
transpose matrix
"""


def matrix_transpose(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_matrix = []
    for row in range(len(matrix[0])):
        new_row = []
        for column in range(len(matrix)):
            new_row.append(matrix[column][row])
        new_matrix.append(new_row)
    return new_matrix
